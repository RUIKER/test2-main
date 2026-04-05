"""
NGAFID 航空数据预处理模块。
包含: 缺失值线性插值填充、防止统计泄露的 Fold-wise Z-score 归一化。
只读取本地 data/subset_data 中已下载的数据，不依赖任何上游链接。
依赖包: pip install numpy pandas scikit-learn compress_pickle
"""
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

try:
    import pyarrow as pa  # type: ignore[reportMissingImports]
    import pyarrow.dataset as pa_ds  # type: ignore[reportMissingImports]
except ModuleNotFoundError:
    pa = None
    pa_ds = None

def load_pickle(file_path):
    try:
        from compress_pickle import load as compress_pickle_load  # pyright: ignore[reportMissingImports]
    except ModuleNotFoundError:
        compress_pickle_load = None

    if compress_pickle_load is not None:
        return compress_pickle_load(file_path)

    with open(file_path, "rb") as file_handle:
        return pickle.load(file_handle)


def _normalize_before_after_label(value) -> int:
    if pd.isna(value):
        raise ValueError("before_after 标签存在空值。")

    if isinstance(value, (np.integer, int)):
        numeric_value = int(value)
        if numeric_value in (0, 1):
            return numeric_value
        raise ValueError(f"不支持的数值标签: {value}")

    text_value = str(value).strip().lower()
    if text_value in {"1", "before", "pre", "prior"}:
        return 1
    if text_value in {"0", "after", "same", "post"}:
        return 0

    raise ValueError(f"不支持的 before_after 标签值: {value}")

class AviationDataPreprocessor:
    def __init__(self):
        """
        初始化预处理器，用于保存当前折(fold)训练集的均值和标准差。
        绝对不使用全局统计量以防数据泄露。
        """
        self.mean_ = None
        self.std_ = None

    def fill_missing_values_linear(self, X: np.ndarray) -> np.ndarray:
        """
        对三维时间序列张量进行缺失值线性插值。
        
        参数:
            X: np.ndarray, 形状为 [samples, timesteps, features]
        返回:
            X_filled: 填充后的 np.ndarray
        """
        print("正在进行特征维度的时间序列线性插值...")
        X_filled = np.empty_like(X)
        # 修复了潜在的元组迭代报错问题，获取样本数量
        n_samples = X.shape[0]
        
        for i in range(n_samples):
            # 将每个样本 [timesteps, features] 转为 DataFrame 处理
            df = pd.DataFrame(X[i])
            # 按列（特征）沿时间轴进行线性插值，limit_direction='both' 防止首尾连续NaN
            df.interpolate(method='linear', limit_direction='both', inplace=True)
            # 对于极端情况（例如某个特征从头到尾全是NaN），使用0兜底
            df.fillna(0, inplace=True)
            X_filled[i] = df.values
            
        return X_filled

    def fit(self, X_train: np.ndarray):
        """
        仅在当前训练折 (Training Fold) 上拟合统计参数 (Mean, Std)。
        
        参数:
            X_train: 形状为 [samples, timesteps, features]
        """
        # 沿样本轴(axis=0)和时间步轴(axis=1)计算，得到长度为 features 的向量
        self.mean_ = np.nanmean(X_train, axis=(0, 1))
        self.std_ = np.nanstd(X_train, axis=(0, 1))
        
        # 防止除零错误 (常数特征)
        self.std_[self.std_ == 0] = 1e-8

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        使用拟合好的训练集参数对输入数据进行 Z-score 归一化。
        
        参数:
            X: 形状为 [samples, timesteps, features] (可以是训练集、验证集或测试集)
        """
        if self.mean_ is None or self.std_ is None:
            raise ValueError("预处理器尚未拟合！请先调用 fit(X_train)。")
        
        return (X - self.mean_) / self.std_

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """拟合并直接转换训练集"""
        self.fit(X_train)
        return self.transform(X_train)


def _resolve_local_subset_dir(
    subset_name: str = "2days",
    base_dir: Path | None = None,
) -> Path:
    project_root = Path(base_dir).resolve() if base_dir is not None else Path(__file__).resolve().parent.parent
    candidates = [
        project_root / "data" / "subset_data" / subset_name / subset_name,
        project_root / "data" / "subset_data" / subset_name,
        project_root / "data" / subset_name / subset_name,
        project_root / "data" / subset_name,
    ]

    required_files = {"flight_data.pkl", "flight_header.csv", "stats.csv"}
    for candidate in candidates:
        if candidate.exists() and required_files.issubset({path.name for path in candidate.iterdir() if path.is_file()}):
            return candidate

    raise FileNotFoundError(
        f"未找到本地数据目录 {subset_name}，请确认数据已放入 data/subset_data/{subset_name}/{subset_name}。"
    )


def load_local_subset_data(
    subset_name: str = "2days",
    base_dir: Path | None = None,
    label_column: str = "before_after",
    max_length: int = 4096,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    从本地 data/subset_data 读取 NGAFID 子集，并构造成预处理所需的三维张量。

    返回:
        X: [samples, timesteps, features]
        y: 标签数组
        flight_header_df: 原始头表，便于后续按 fold 或其它列筛选
    """
    subset_dir = _resolve_local_subset_dir(subset_name=subset_name, base_dir=base_dir)
    flight_header_path = subset_dir / "flight_header.csv"
    flight_data_path = subset_dir / "flight_data.pkl"

    flight_header_df = pd.read_csv(flight_header_path, index_col="Master Index")
    flight_data_dict = load_pickle(flight_data_path)

    if label_column not in flight_header_df.columns:
        raise KeyError(f"标签列不存在: {label_column}")

    sample_ids = [index for index in flight_header_df.index if index in flight_data_dict]
    if not sample_ids:
        raise ValueError("flight_header.csv 中的索引在 flight_data.pkl 中都未找到。")

    if max_samples is not None and max_samples > 0 and len(sample_ids) > max_samples:
        rng = np.random.default_rng(42)
        sample_ids = list(rng.choice(sample_ids, size=max_samples, replace=False))
        print(f"样本量过大，已按固定随机种子下采样到 {len(sample_ids)} 条用于训练。")

    first_sample = np.asarray(flight_data_dict[sample_ids[0]], dtype=np.float32)
    feature_count = first_sample.shape[1]
    max_length = min(max_length, max(np.asarray(flight_data_dict[index]).shape[0] for index in sample_ids))

    X = np.zeros((len(sample_ids), max_length, feature_count), dtype=np.float32)
    y = np.zeros(len(sample_ids), dtype=np.int64)

    for i, index in enumerate(sample_ids):
        sample_array = np.asarray(flight_data_dict[index], dtype=np.float32)
        sample_array = sample_array[-max_length:, :feature_count]
        X[i, : sample_array.shape[0], :] = sample_array
        y[i] = _normalize_before_after_label(flight_header_df.loc[index, label_column])

    return X, y, flight_header_df.loc[sample_ids]


def _resolve_all_flights_dir(base_dir: Path | None = None) -> Path:
    project_root = Path(base_dir).resolve() if base_dir is not None else Path(__file__).resolve().parent.parent
    candidates = [
        project_root / "data" / "subset_data" / "all_flights",
        project_root / "data" / "subset_data" / "all_flight" / "all_flights",
    ]

    for candidate in candidates:
        if candidate.exists() and (candidate / "flight_header.csv").exists() and (candidate / "one_parq").exists():
            return candidate

    raise FileNotFoundError("未找到 all_flights 数据目录，请先下载并解压 all_flights 到 data/subset_data。")


def load_all_flights_data(
    base_dir: Path | None = None,
    label_column: str = "before_after",
    max_length: int = 1024,
    max_samples: int = 300,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    从 all_flights/one_parq 构建训练张量。
    说明：该过程成本较高，默认进行采样以避免内存和CPU过载。
    """
    all_flights_dir = _resolve_all_flights_dir(base_dir=base_dir)
    header_path = all_flights_dir / "flight_header.csv"
    one_parq_path = all_flights_dir / "one_parq"

    header_df = pd.read_csv(header_path, index_col="Master Index")
    if label_column not in header_df.columns:
        raise KeyError(f"all_flights 缺少标签列: {label_column}")

    sample_ids = list(header_df.index)
    if max_samples > 0 and len(sample_ids) > max_samples:
        rng = np.random.default_rng(42)
        sample_ids = list(rng.choice(sample_ids, size=max_samples, replace=False))

    if pa_ds is None or pa is None:
        raise RuntimeError("缺少 pyarrow，无法按需读取 all_flights parquet。")

    dataset = pa_ds.dataset(str(one_parq_path), format="parquet")
    schema_names = dataset.schema.names

    if "Master Index" in schema_names:
        id_col = "Master Index"
    elif "Index" in schema_names:
        id_col = "Index"
    else:
        id_col = None

    if id_col is None:
        raise ValueError("all_flights parquet 中找不到 flight id 列。")

    time_col = "timestep" if "timestep" in schema_names else None
    excluded_cols = {id_col, "timestep", label_column}
    feature_cols = []
    for field in dataset.schema:
        if field.name in excluded_cols:
            continue
        if pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
            feature_cols.append(field.name)

    if not feature_cols:
        raise ValueError("all_flights parquet 中未找到可用数值特征列。")

    selected_columns = [id_col]
    if time_col is not None:
        selected_columns.append(time_col)
    selected_columns.extend(feature_cols)

    try:
        table = dataset.to_table(
            filter=pa_ds.field(id_col).isin(sample_ids),
            columns=selected_columns,
        )
    except Exception as exc:
        raise RuntimeError(f"读取 all_flights parquet 失败: {exc}") from exc

    if table.num_rows == 0:
        raise ValueError("all_flights parquet 过滤后为空，无法构建训练数据。")

    working_df = table.to_pandas()
    grouped = working_df.groupby(id_col, sort=False)
    usable_ids = [fid for fid in sample_ids if fid in grouped.groups]
    if not usable_ids:
        raise ValueError("all_flights 中找不到与 flight_header 对应的航班序列。")

    X = np.zeros((len(usable_ids), max_length, len(feature_cols)), dtype=np.float32)
    y = np.zeros(len(usable_ids), dtype=np.int64)

    for i, fid in enumerate(usable_ids):
        flight_df = grouped.get_group(fid)
        if time_col is not None:
            flight_df = flight_df.sort_values(time_col)
        arr = flight_df[feature_cols].to_numpy(dtype=np.float32)
        arr = arr[-max_length:, :]
        X[i, :arr.shape[0], :] = arr
        y[i] = _normalize_before_after_label(header_df.loc[fid, label_column])

    return X, y, header_df.loc[usable_ids]


def load_combined_training_data(
    base_dir: Path | None = None,
    label_column: str = "before_after",
    max_length: int = 1024,
    max_samples_2days: int = 1500,
    max_samples_all_flights: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """同时读取 2days 和 all_flights，并拼接为统一训练集。"""
    X_2days, y_2days, _ = load_local_subset_data(
        subset_name="2days",
        base_dir=base_dir,
        label_column=label_column,
        max_length=max_length,
        max_samples=max_samples_2days,
    )

    X_all, y_all, _ = load_all_flights_data(
        base_dir=base_dir,
        label_column=label_column,
        max_length=max_length,
        max_samples=max_samples_all_flights,
    )

    feature_count = min(X_2days.shape[2], X_all.shape[2])
    if X_2days.shape[2] != X_all.shape[2]:
        print(f"特征维度不一致，按最小公共维度裁剪: 2days={X_2days.shape[2]}, all_flights={X_all.shape[2]}, used={feature_count}")

    X_merged = np.concatenate([
        X_2days[:, :, :feature_count],
        X_all[:, :, :feature_count],
    ], axis=0)
    y_merged = np.concatenate([y_2days.astype(np.int64), y_all.astype(np.int64)], axis=0)

    print(f"合并数据完成: 2days={X_2days.shape[0]} 条, all_flights={X_all.shape[0]} 条, total={X_merged.shape[0]} 条")
    return X_merged, y_merged


def format_labels(y_raw: np.ndarray) -> np.ndarray:
    """
    根据任务需求对齐标签：
    确保 0 = 维护后 (健康航班)
    确保 1 = 维护前 (即将发生故障的航班)
    """
    y_formatted = y_raw.copy()
    return y_formatted


# ================= 使用范例（仅本地 data） ================= 
def run_local_cv_example():
    """演示数据隔离的正确做法：每折都独立拟合统计量，不泄露全数据统计"""
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    
    print("正在从本地 data/subset_data 读取 2days 数据...")
    dummy_X, dummy_y, _ = load_local_subset_data(subset_name="2days", label_column="before_after")

    # 统一转换标签
    dummy_y = format_labels(dummy_y)

    # 声明分层交叉验证 (random_state 固定确保可复现)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(dummy_X, dummy_y)):
        print(f"\n--- 正在处理第 {fold + 1} 折 ---")
        X_train, y_train = dummy_X[train_idx], dummy_y[train_idx]
        X_val, y_val = dummy_X[val_idx], dummy_y[val_idx]
        
        # 每折都新建预处理器 (最关键的防泄露步骤)
        preprocessor = AviationDataPreprocessor()
        
        # 1. 对训练集和验证集分别进行插值 (sample-wise 独立，无统计泄露)
        X_train = preprocessor.fill_missing_values_linear(X_train)
        X_val = preprocessor.fill_missing_values_linear(X_val)
        
        # 2. 严格防止泄漏的归一化
        # fit_transform 只在训练集上拟合 mean/std，然后用这些统计量转换验证集
        # 这样验证集的统计特性完全来自训练集定义，不泄露验证集本身的信息
        X_train_scaled = preprocessor.fit_transform(X_train)
        X_val_scaled = preprocessor.transform(X_val)
        
        print(f"X_train 形状: {X_train_scaled.shape}, Mean 验证 (应接近0): {np.mean(X_train_scaled):.4f}")
        print(f"X_val 形状: {X_val_scaled.shape}")


def run_cv_example():
    """兼容旧入口：改为直接使用本地数据。"""
    run_local_cv_example()

if __name__ == "__main__":
    run_local_cv_example()