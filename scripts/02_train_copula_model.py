import os
import sys
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import DATA_PATHS
from src.data_loader import get_nhts_trip_data
from src.trip_modeling import build_marginal_distributions
from src.copula_model import SpatiotemporalDependenceModel


def main():
    print("=== [02_train_copula_model] 启动 Frank Copula 参数训练 ===")

    output_dir = os.path.join(project_root, "results", "tables")
    os.makedirs(output_dir, exist_ok=True)

    nhts_path = DATA_PATHS.get("nhts_trip_data")
    print(f"[*] 读取 NHTS 原始数据: {nhts_path}")
    
    try:
        df_trips = get_nhts_trip_data(nhts_path)
    except FileNotFoundError as e:
        print(f"\n[Fatal Error] {e}")
        print("执行中断: 联合分布训练缺失底层时空成对数据。")
        sys.exit(1)

    # 剔除无效值，确保联合分布样本成对严格对齐
    valid_mask = (df_trips['Departure_Hour'] >= 0) & (df_trips['Trip_Distance_Miles'] >= 0)
    df_valid_trips = df_trips[valid_mask].copy()
    
    raw_times = df_valid_trips['Departure_Hour'].values
    raw_distances = df_valid_trips['Trip_Distance_Miles'].values
    print(f"[*] 成功提取 {len(raw_times)} 对有效时空样本。")

    print("[*] 拟合边缘分布 (PMF & KDE)...")
    dist_results = build_marginal_distributions(df_valid_trips)
    time_pmf = dist_results['time_pmf']
    distance_kde = dist_results['distance_kde']

    print("[*] 初始化 SpatiotemporalDependenceModel 引擎...")
    copula_engine = SpatiotemporalDependenceModel(time_pmf=time_pmf, distance_kde=distance_kde, theta=-1.0)

    print("[*] 执行 MLE 优化计算最优 theta...")
    try:
        best_theta = copula_engine.train_copula_parameter(raw_times, raw_distances)
        print(f"[+] MLE 拟合收敛。最优参数 theta = {best_theta:.4f}")
    except Exception as e:
        print(f"\n[Fatal Error] MLE 拟合失败: {e}")
        sys.exit(1)

    params_output_path = os.path.join(output_dir, "copula_trained_parameters.json")
    
    trained_params = {
        "model": "Frank Copula",
        "sample_size": len(raw_times),
        "theta": best_theta,
        "kde_bandwidth": float(distance_kde.bandwidth),
    }

    try:
        with open(params_output_path, "w", encoding="utf-8") as f:
            json.dump(trained_params, f, indent=4, ensure_ascii=False)
        print(f"[+] 训练参数已固化至: {params_output_path}")
    except Exception as e:
        print(f"\n[Warning] 参数固化失败: {e}")

    print("=== [02_train_copula_model] 执行完毕 ===\n")


if __name__ == "__main__":
    main()
