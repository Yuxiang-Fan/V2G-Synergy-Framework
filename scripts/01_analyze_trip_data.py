"""
执行脚本 01: 出行时空特征分析 (01_analyze_trip_data.py)
读取外部 NHTS 出行链数据，拟合出发时间与行驶距离的边缘分布，并导出统计参数。
"""

import os
import sys
import pandas as pd

# 动态挂载项目根目录到 sys.path，确保能够正确导入 src 模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入核心功能模块与配置
from src.config import DATA_PATHS
from src.data_loader import get_nhts_trip_data
from src.trip_modeling import build_marginal_distributions


def main():
    print("=== 开始执行: 01_analyze_trip_data ===")

    # 1. 确保输出目录存在
    output_dir = os.path.join(project_root, "results", "tables")
    os.makedirs(output_dir, exist_ok=True)

    # 2. 读取外部 NHTS 数据 (严格占位，依赖外部数据文件真实存在)
    nhts_path = DATA_PATHS.get("nhts_trip_data")
    print(f"正在从外部路径读取 NHTS 原始数据: {nhts_path}")
    
    try:
        df_trips = get_nhts_trip_data(nhts_path)
    except FileNotFoundError as e:
        print(f"\n[错误] {e}")
        print("请确保已下载开源 NHTS 数据，并正确配置了 src/config.py 中的 DATA_PATHS。")
        sys.exit(1)

    print(f"成功加载 {len(df_trips)} 条有效出行记录。")

    # 3. 构建边缘分布与提取统计特征
    print("正在拟合出发时间 PMF (经验概率质量函数) 与行驶距离 KDE (核密度估计)...")
    dist_results = build_marginal_distributions(df_trips)

    time_pmf = dist_results['time_pmf']
    stats = dist_results['statistics']
    kde_bandwidth = dist_results['distance_kde'].bandwidth

    # 4. 打印核心发现 (直接对应论文 2.2.1 节的内容)
    print("\n【出行距离统计特征计算结果】")
    print(f"样本总数: {stats['count']}")
    print(f"平均距离: {stats['mean']:.2f} 英里")
    print(f"中位数距离: {stats['median']:.2f} 英里")
    print(f"距离 <= 5 英里占比: {stats['pct_under_5_miles']:.1f}%")
    print(f"距离 5~30 英里占比: {stats['pct_5_to_30_miles']:.1f}%")
    print(f"距离 > 30 英里占比: {stats['pct_over_30_miles']:.1f}%")
    print(f"KDE 自适应平滑带宽 (Silverman): {kde_bandwidth:.4f}")

    # 5. 导出拟合的概率参数以供后续模块或分析使用
    output_pmf_path = os.path.join(output_dir, "departure_time_pmf.csv")
    df_pmf = pd.DataFrame({
        'Hour': range(24),
        'Probability': time_pmf
    })
    
    try:
        df_pmf.to_csv(output_pmf_path, index=False)
        print(f"\n✅ 出发时间概率分布已成功导出至: {output_pmf_path}")
    except Exception as e:
        print(f"\n[警告] 导出结果时发生错误: {e}")

    print("=== 01_analyze_trip_data 执行完毕 ===\n")


if __name__ == "__main__":
    main()