"""
航空预见性维护二分类模型训练与评估脚本
模型: MiniRocketMultivariate (轻量级时间序列SOTA)
验证: 分层交叉验证 (严格防止数据泄露)
依赖包: pip install numpy pandas scikit-learn matplotlib seaborn sktime numba
"""

import os
import time
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm


def setup_numba_cache_dir():
    """将 numba 缓存目录指向项目内可写路径，避免写入 site-packages 失败。"""
    if os.environ.get("NUMBA_CACHE_DIR"):
        return
    cache_dir = Path(__file__).resolve().parent / ".numba_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)


setup_numba_cache_dir()
from sktime.transformations.panel.rocket import MiniRocketMultivariate


# ================= 1. 环境与中文字体配置 =================
def setup_matplotlib():
    """配置 matplotlib 以兼容所有操作系统支持中文显示并正常显示负号"""
    system = platform.system()
    if system == "Windows":
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    elif system == "Darwin":
        plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
    else:
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


# ================= 2. 严格防泄露的预处理器 =================
class FoldPreprocessor:
    def __init__(self):
        """每折必须重新实例化，用于保存当前训练集的统计量"""
        self.mean_ = None
        self.std_ = None

    def fill_missing(self, X: np.ndarray, progress_desc: str | None = None) -> np.ndarray:
        """特征维度时间序列线性插值 (独立对每个样本进行，不产生泄露)"""
        X_filled = np.empty_like(X)
        sample_iter = range(X.shape[0])
        if progress_desc is not None:
            sample_iter = tqdm(
                sample_iter,
                total=X.shape[0],
                desc=progress_desc,
                unit="sample",
                leave=False,
                dynamic_ncols=True,
            )

        for i in sample_iter:
            df = pd.DataFrame(X[i])
            df.interpolate(method="linear", limit_direction="both", inplace=True)
            df.fillna(0, inplace=True)
            X_filled[i] = df.values
        return X_filled

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """仅在训练集上拟合统计量并转换"""
        self.mean_ = np.nanmean(X_train, axis=(0, 1))
        self.std_ = np.nanstd(X_train, axis=(0, 1))
        self.std_[self.std_ == 0] = 1e-8
        return (X_train - self.mean_) / self.std_

    def transform(self, X_test: np.ndarray) -> np.ndarray:
        """用训练集的统计量转换测试集"""
        return (X_test - self.mean_) / self.std_


# ================= 3. 核心训练与评估流水线 =================
def train_and_evaluate(X: np.ndarray, y: np.ndarray):
    """
    执行交叉验证并保存结果。
    参数:
        X: 形状为 [samples, timesteps, features] 的原始数据
        y: 二元标签 (0: 健康/维护后, 1: 故障/维护前)
    """
    random_seed = 42
    np.random.seed(random_seed)

    y = np.asarray(y)
    try:
        y = y.astype(np.int64)
    except Exception as exc:
        raise TypeError(f"标签必须是可转为整数的数值类型，当前 dtype={y.dtype}") from exc

    n_splits = 5

    setup_matplotlib()

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics = {"accuracy": [], "f1_weighted": [], "roc_auc": []}
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    aggregate_cm = np.zeros((2, 2))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    print(f"数据总形状: X={X.shape}, y={y.shape}")
    print(f"标签分布: {np.bincount(y)}")
    print("说明: 第1折通常最慢（MiniRocket/Numba 首次编译），后续折会明显加速。")
    print(f"开始 {n_splits} 折交叉验证 (模型: MiniRocketMultivariate)...\n")

    start_total_time = time.time()

    fold_iter = tqdm(
        skf.split(X, y),
        total=n_splits,
        desc="交叉验证总进度",
        unit="fold",
        dynamic_ncols=True,
    )

    for fold, (train_idx, test_idx) in enumerate(fold_iter, start=1):
        print(f"[Fold] 正在运行第 {fold} 折交叉验证...")
        fold_start = time.time()
        fold_stage_bar = tqdm(
            total=4,
            desc=f"Fold {fold}/{n_splits} 阶段",
            unit="step",
            leave=False,
            dynamic_ncols=True,
        )

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        preprocessor = FoldPreprocessor()
        X_train_filled = preprocessor.fill_missing(X_train, progress_desc=f"Fold {fold}/{n_splits} 训练集插值")
        X_test_filled = preprocessor.fill_missing(X_test, progress_desc=f"Fold {fold}/{n_splits} 测试集插值")

        X_train_scaled = preprocessor.fit_transform(X_train_filled)
        X_test_scaled = preprocessor.transform(X_test_filled)
        fold_stage_bar.update(1)
        print(f"  - 预处理完成，用时 {time.time() - fold_start:.2f}s")

        X_train_sktime = np.transpose(X_train_scaled, (0, 2, 1))
        X_test_sktime = np.transpose(X_test_scaled, (0, 2, 1))

        minirocket = MiniRocketMultivariate(random_state=random_seed)
        X_train_features = minirocket.fit_transform(X_train_sktime)
        X_test_features = minirocket.transform(X_test_sktime)
        fold_stage_bar.update(1)
        print(f"  - 特征提取完成，用时 {time.time() - fold_start:.2f}s")

        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        classifier.fit(X_train_features, y_train)
        fold_stage_bar.update(1)

        y_pred = classifier.predict(X_test_features)
        y_score = classifier.decision_function(X_test_features)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        auc_score = roc_auc_score(y_test, y_score)

        metrics["accuracy"].append(acc)
        metrics["f1_weighted"].append(f1)
        metrics["roc_auc"].append(auc_score)
        aggregate_cm += confusion_matrix(y_test, y_pred)

        fpr, tpr, _ = roc_curve(y_test, y_score)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0
        tprs.append(interp_tpr)
        fold_stage_bar.update(1)
        fold_stage_bar.close()

        print(f"  -> 结果: Accuracy={acc:.4f} | F1={f1:.4f} | AUC={auc_score:.4f}")
        print(f"  - 第 {fold} 折总耗时: {time.time() - fold_start:.2f}s\n")

    print("========================================")
    print(f"{n_splits} 折交叉验证全部完成！总耗时: {time.time() - start_total_time:.2f}s")

    print("\n【最终评估指标 (均值 ± 标准差)】")
    print(f"Accuracy:    {np.mean(metrics['accuracy']):.4f} ± {np.std(metrics['accuracy']):.4f}")
    print(f"Weighted F1: {np.mean(metrics['f1_weighted']):.4f} ± {np.std(metrics['f1_weighted']):.4f}")
    print(f"ROC-AUC:     {np.mean(metrics['roc_auc']):.4f} ± {np.std(metrics['roc_auc']):.4f}")
    print("\n* 指标说明: 在航空故障这类潜在的不平衡数据中，单独的 Accuracy 极易产生掩蔽效应（将少数类全判为多数类依然有高准确率）。")
    print("* Weighted F1 综合了精确率和召回率，而 ROC-AUC 衡量了模型在不同判定阈值下对正负样本的排序区分能力，两者能更客观地反映模型抓取真正故障(维护前)特征的能力。")

    plt.figure(figsize=(6, 5))
    plt.imshow(aggregate_cm, cmap="Blues")
    plt.colorbar(label="样本数")
    plt.xticks([0, 1], ["维护后(健康)", "维护前(故障)"])
    plt.yticks([0, 1], ["维护后(健康)", "维护前(故障)"])
    for i in range(aggregate_cm.shape[0]):
        for j in range(aggregate_cm.shape[1]):
            plt.text(j, i, f"{aggregate_cm[i, j]:.0f}", ha="center", va="center", color="black")
    plt.title(f"{n_splits}折交叉验证累计混淆矩阵 (MiniRocket)")
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 6))
    for i in range(len(tprs)):
        plt.plot(mean_fpr, tprs[i], alpha=0.3, label=f"第 {i+1} 折 (AUC = {metrics['roc_auc'][i]:.3f})")

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(metrics["roc_auc"])

    plt.plot(mean_fpr, mean_tpr, color="b", label=f"平均 ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})", lw=2, alpha=0.8)
    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="随机猜测", alpha=0.8)
    plt.title("受试者工作特征曲线 (ROC-AUC)")
    plt.xlabel("假阳性率 (False Positive Rate)")
    plt.ylabel("真阳性率 (True Positive Rate)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(results_dir / "roc_curve.png", dpi=300)
    plt.close()

    folds = np.arange(1, len(metrics["accuracy"]) + 1)
    width = 0.25
    plt.figure(figsize=(8, 5))
    plt.bar(folds - width, metrics["accuracy"], width, label="Accuracy")
    plt.bar(folds, metrics["f1_weighted"], width, label="Weighted F1")
    plt.bar(folds + width, metrics["roc_auc"], width, label="ROC-AUC")

    plt.title("MiniRocket 各折验证集性能对比")
    plt.xlabel("交叉验证折数 (Fold)")
    plt.ylabel("Score")
    plt.ylim(0, 1.1)
    plt.xticks(folds)
    plt.legend(loc="lower right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(results_dir / "fold_metrics_comparison.png", dpi=300)
    plt.close()

    print(f"\n所有评估图表已成功保存至: {results_dir.absolute()}")


if __name__ == "__main__":
    print("生成并加载测试数据集 (格式: [样本数, 4096, 23])...")
    num_samples = 200
    np.random.seed(42)
    dummy_X = np.random.randn(num_samples, 4096, 23)
    dummy_X[1, 100:200, 5] = np.nan
    dummy_y = np.random.randint(0, 2, num_samples)

    train_and_evaluate(dummy_X, dummy_y)
