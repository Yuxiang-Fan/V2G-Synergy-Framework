import os
import sys
import time
import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import (
    TARGET_CITIES, 
    PARTICIPATION_RATES, 
    NUM_SIMULATION_RUNS, 
    NUM_PROCESSES, 
    RANDOM_SEED
)
from src.data_loader import load_city_natural_load, get_city_ev_count
from src.load_analysis import build_grid_context
from src.v2g_strategy import run_parallel_simulations
from src.evaluation import calculate_evaluation_metrics


def generate_city_bt_array(total_ev_count: int) -> np.ndarray:
    """基于先验调研比例，构造全城 EV 的 BT 物理参数映射数组"""
    base_ratios = np.array([0.004, 0.121, 0.568, 0.144, 0.018, 0.117, 0.003, 0.025])
    base_ratios = base_ratios / np.sum(base_ratios)
    counts = np.round(total_ev_count * base_ratios).astype(int)
    counts[2] += total_ev_count - counts.sum()
    
    bt_list = []
    for bt_idx, cnt in enumerate(counts, start=1):
        bt_list.extend([bt_idx] * cnt)
        
    np.random.shuffle(bt_list)
    return np.array(bt_list)


def main():
    print("=== [04_run_v2g_simulation] 启动多区域 V2G 协同调度仿真 ===")
    
    output_dir = os.path.join(project_root, "results", "tables")
    os.makedirs(output_dir, exist_ok=True)
    
    natural_load_dir = os.path.join(project_root, "results", "natural_loads")
    if not os.path.exists(natural_load_dir):
        print(f"[Fatal Error] 缺失纯净负荷基线 Lnatural: {natural_load_dir}")
        print("请确认已优先执行 03_run_baseline_load.py。")
        sys.exit(1)
        
    fallback_ev_counts = {
        "San Diego": 99183, "San Francisco": 28307, "Los Angeles": 282829,
        "New York": 189440, "Houston": 92519
    }
    
    all_cities_metrics = []

    for city in TARGET_CITIES:
        print(f"\n{'='*50}\n[*] 启动城市级 V2G 仿真: {city}\n{'='*50}")
        
        try:
            df_natural = load_city_natural_load(city, base_dir=natural_load_dir)
            natural_load = df_natural['Lnatural'].values
        except FileNotFoundError as e:
            print(f"[Skip] {e}")
            continue
            
        try:
            ev_count = get_city_ev_count(city, zip_codes=[])
        except FileNotFoundError:
            ev_count = fallback_ev_counts.get(city, 0)
            
        if ev_count <= 0:
            print(f"[Skip] 城市 {city} 车辆注册数据异常。")
            continue
            
        bt_array = generate_city_bt_array(ev_count)
        print(f"[+] 物理模型构建完毕: 注入 {ev_count} 辆 EV。")
        
        # 构建电网感知 Context，识别峰谷时段并计算权重
        grid_ctx = build_grid_context(df_natural, load_col='Lnatural')
        print(f"[+] 电网感知就绪 | 峰区: {grid_ctx['peak_hours']} | 谷区: {grid_ctx['valley_hours']}")

        df_all_loads = pd.DataFrame({'Hour': range(24), 'Lnatural': natural_load})
        city_metrics_list = []
        
        start_city_time = time.time()
        
        for pr in PARTICIPATION_RATES:
            pr_label = f"{pr*100:.0f}%"
            print(f"\n--- 注入 V2G 参与度: {pr_label} | 并行迭代: {NUM_SIMULATION_RUNS} 次 ---")
            
            # 核心引擎：多进程执行蒙特卡洛调度，获取净负荷增量
            net_dispatch_load = run_parallel_simulations(
                participation_rate=pr,
                grid_ctx=grid_ctx,
                bt_array=bt_array,
                total_runs=NUM_SIMULATION_RUNS,
                num_processes=NUM_PROCESSES,
                random_seed=RANDOM_SEED
            )
            
            # 合成最终 Ldispatch
            dispatch_load = natural_load + net_dispatch_load
            df_all_loads[f'Ldispatch_{pr_label}'] = dispatch_load
            
            # 触发指标评估模块
            metrics = calculate_evaluation_metrics(natural_load, dispatch_load)
            metrics.update({
                'City': city,
                'Participation_Rate': pr,
                'EV_Count': ev_count
            })
            city_metrics_list.append(metrics)
            
            print(f"    -> 削峰率: {metrics['Peak_Reduction_Rate_%']:.2f}% | "
                  f"负荷率: {metrics['Load_Ratio_Dispatch_%']:.2f}% | "
                  f"峰谷差: {metrics['Delta_L_Dispatch']:.0f} kW")

        elapsed = time.time() - start_city_time
        print(f"\n🎉 城市 {city} 全参量空间仿真收敛，耗时 {elapsed:.2f} s。")

        # 持久化该城市的负荷曲线阵列与指标发展轨迹
        formatted_city = city.replace(" ", "_")
        load_output_path = os.path.join(output_dir, f"{formatted_city}_V2G_Load_Curves.xlsx")
        metrics_output_path = os.path.join(output_dir, f"{formatted_city}_V2G_Metrics.csv")
        
        try:
            df_all_loads.to_excel(load_output_path, index=False)
            pd.DataFrame(city_metrics_list).to_csv(metrics_output_path, index=False)
            print(f"📄 结果落盘: {formatted_city}_V2G_Load_Curves.xlsx & _Metrics.csv")
        except Exception as e:
            print(f"[Error] 数据落盘失败: {e}")
            
        all_cities_metrics.extend(city_metrics_list)

    if all_cities_metrics:
        global_metrics_path = os.path.join(output_dir, "All_Cities_V2G_Metrics_Summary.csv")
        try:
            pd.DataFrame(all_cities_metrics).to_csv(global_metrics_path, index=False)
            print(f"\n✅ 跨区域全局评估指标汇总完毕: {global_metrics_path}")
        except Exception as e:
            print(f"[Error] 全局汇总表导出失败: {e}")

    print("\n=== [04_run_v2g_simulation] 执行完毕 ===")


if __name__ == "__main__":
    main()
