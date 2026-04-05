"""
航空预见性维护项目 - 自动化全流程入口
"""
import os
import sys
import traceback
from pathlib import Path

# 将 src 目录动态加入环境变量，确保兼容所有操作系统
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir / "src"))

from data_downloader import extract_and_download_subset
from data_preprocessor import load_local_subset_data, format_labels
from train_evaluate import train_and_evaluate

def load_data(project_root: Path):
    """
    数据加载中枢：
    读取本地子集数据（首次运行会先触发自动下载）。
    """
    max_length = int(os.environ.get("PM_MAX_LENGTH", "4096"))

    print("正在读取本地数据: 2days ...")
    print(f"加载参数: max_length={max_length}, max_samples=全量")
    X, y, _ = load_local_subset_data(
        base_dir=project_root,
        label_column="before_after",
        max_length=max_length,
    )
    y = format_labels(y)
    print(f"数据加载完成: X={X.shape}, y={y.shape}")
    return X, y

def main():
    print("=" * 60)
    print(" ✈️ 航空预见性维护 (Predictive Maintenance) 自动化流水线")
    print("=" * 60)
    
    try:
        # [步骤 1] 自动解析并拉取数据
        print("\n>>> [阶段 1/3] 检查并拉取 NGAFID 基准数据集...")
        extract_and_download_subset()
        
        # [步骤 2] 加载时间序列数据
        print("\n>>> [阶段 2/3] 加载多变量时间序列数据...")
        X, y = load_data(current_dir)
        
        # [步骤 3] 严格防泄露的 5 折交叉验证与模型训练
        print("\n>>> [阶段 3/3] 启动 MiniRocket 模型训练与性能评估...")
        train_and_evaluate(X, y)
        
        print("\n" + "=" * 60)
        print("✅ 全流程执行完毕！请前往 `results/` 目录查看：")
        print("   - confusion_matrix.png (混淆矩阵)")
        print("   - roc_curve.png (ROC-AUC 曲线)")
        print("   - fold_metrics_comparison.png (各折指标对比图)")
        print("=" * 60)
        
    except Exception as e:
        print("\n❌ 运行过程中发生严重异常，流水线已终止！")
        print("=" * 60)
        traceback.print_exc()
        print("=" * 60)
        print("【💡 错误排查指南】")
        print("1. 如果是依赖报错：请确认是否已执行 `pip install -r requirements.txt`。")
        print("2. 如果是下载失败：请检查当前网络是否可访问 Zenodo/Google Drive。")
        print("3. 如果是数据读取失败：请确认 `data/subset_data` 下存在 2days 数据。")

if __name__ == "__main__":
    main()