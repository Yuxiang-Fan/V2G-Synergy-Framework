"""
执行脚本 02: 训练 Copula 联合分布模型 (02_train_copula_model.py)
基于外部真实的 NHTS 出行时空数据，使用最大似然估计 (MLE) 计算 Frank Copula 的核心依赖参数 theta，
并将其保存以供后续蒙特卡洛 V2G 规模化抽样使用。
"""

import os
import sys
import json
import pandas as pd

# 动态挂载项目根目录到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import DATA_PATHS
from src.data_loader import get_nhts_trip_data
from src.trip_modeling import build_marginal_distributions
from src.copula_model import SpatiotemporalDependenceModel


def main():
    print("=== 开始执行: 02_train_copula_model ===")

    # 1. 检查并准备输出目录
    output_dir = os.path.join(project_root, "results", "tables")
    os.makedirs(output_dir, exist_ok=True)

    # 2. 读取真实的 NHTS 出行链外部数据 (严格占位)
    nhts_path = DATA_PATHS.get("nhts_trip_data")
    print(f"正在读取外部 NHTS 原始数据: {nhts_path}")
    
    try:
        df_trips = get_nhts_trip_data(nhts_path)
    except FileNotFoundError as e:
        print(f"\n[错误] {e}")
        print("执行中断: Copula 模型的联合分布训练必须依赖真实的二维时空成对数据。")
        sys.exit(1)

    # 剔除无效负值，确保时间和距离对齐 (成对出现才能计算联合分布)
    valid_mask = (df_trips['Departure_Hour'] >= 0) & (df_trips['Trip_Distance_Miles'] >= 0)
    df_valid_trips = df_trips[valid_mask].copy()
    
    raw_times = df_valid_trips['Departure_Hour'].values
    raw_distances = df_valid_trips['Trip_Distance_Miles'].values

    # 3. 提取边缘分布作为 Copula 的输入 (U 和 V 的转换基础)
    print("正在构建边缘概率累积分布 (CDF)...")
    dist_results = build_marginal_distributions(df_valid_trips)
    time_pmf = dist_results['time_pmf']
    distance_kde = dist_results['distance_kde']

    # 4. 初始化 Frank Copula 时空依赖模型
    print("初始化 SpatiotemporalDependenceModel 联合分布模型引擎...")
    # 初始传入一个随意的值，随后用 MLE 进行重新拟合
    copula_engine = SpatiotemporalDependenceModel(time_pmf=time_pmf, distance_kde=distance_kde, theta=-1.0)

    # 5. 执行最大似然估计 (MLE) 拟合最佳 theta
    print(f"正在基于 {len(raw_times)} 条真实出行记录执行最大似然估计 (MLE) 计算 theta...")
    try:
        best_theta = copula_engine.train_copula_parameter(raw_times, raw_distances)
        print(f"✅ MLE 参数拟合成功！最优 Frank Copula 参数 theta = {best_theta:.4f}")
        # 备注：依据论文研究，该值预期在 -0.637 左右，反映微弱的负相关关系。
    except Exception as e:
        print(f"\n[错误] MLE 拟合失败: {e}")
        sys.exit(1)

    # 6. 将训练好的核心参数保存，以便供 ev_simulation.py 和 v2g_strategy.py 调用
    # (在实际运行 04_run_v2g_simulation.py 时，可以通过读取此配置文件获取真实的 theta)
    params_output_path = os.path.join(output_dir, "copula_trained_parameters.json")
    
    trained_params = {
        "model": "Frank Copula",
        "sample_size": len(raw_times),
        "theta": best_theta,
        "kde_bandwidth": float(distance_kde.bandwidth),
        "note": "该参数用于在 V2G 模拟中保持出行时间与行驶距离的时空耦合关联性"
    }

    try:
        with open(params_output_path, "w", encoding="utf-8") as f:
            json.dump(trained_params, f, indent=4, ensure_ascii=False)
        print(f"训练参数已固化并保存至: {params_output_path}")
    except Exception as e:
        print(f"\n[警告] 保存 JSON 参数时出错: {e}")

    print("=== 02_train_copula_model 执行完毕 ===\n")


if __name__ == "__main__":
    main()