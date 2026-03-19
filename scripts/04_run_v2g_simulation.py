"""
执行脚本 04: V2G 多参与度有序充电协同仿真 (04_run_v2g_simulation.py)
读取各城市的自然负荷基线，划分峰谷区间并计算概率，
在不同用户参与度下执行大规模多进程 V2G 调度模拟，并计算评估指标。
"""

import os
import sys
import numpy as np
import pandas as pd
import time

# 动态挂载项目根目录到 sys.path
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
    """
    根据给定的总保有量，生成车辆 BT 等级数组 (保持与 03 脚本一致)。
    """
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
    print("=== 开始执行: 04_run_v2g_simulation ===")
    
    # 1. 检查并准备输出目录
    output_dir = os.path.join(project_root, "results", "tables")
    os.makedirs(output_dir, exist_ok=True)
    
    natural_load_dir = os.path.join(project_root, "results", "natural_loads")
    if not os.path.exists(natural_load_dir):
        print(f"[错误] 找不到自然负荷基线目录: {natural_load_dir}")
        print("请先成功运行 03_run_baseline_load.py 提取城市基础负荷。")
        sys.exit(1)
        
    fallback_ev_counts = {
        "San Diego": 99183, "San Francisco": 28307, "Los Angeles": 282829,
        "New York": 189440, "Houston": 92519
    }
    
    # 记录所有城市的总评估结果，用于最终的双因子回归分析
    all_cities_metrics = []

    # 2. 遍历每一个目标城市执行策略模拟
    for city in TARGET_CITIES:
        print(f"\n" + "="*50)
        print(f"正在对城市进行 V2G 规模化仿真: {city}")
        print("="*50)
        
        # 2.1 加载该城市的自然电力负荷 (Lnatural)
        try:
            df_natural = load_city_natural_load(city, base_dir=natural_load_dir)
            natural_load = df_natural['Lnatural'].values
        except FileNotFoundError as e:
            print(f"[跳过] {e}")
            continue
            
        # 2.2 获取电动汽车保有量并生成 BT 数组
        try:
            ev_count = get_city_ev_count(city, zip_codes=[])
        except FileNotFoundError:
            ev_count = fallback_ev_counts.get(city, 0)
            
        if ev_count <= 0:
            print(f"[跳过] 城市 {city} 车辆数据异常。")
            continue
            
        bt_array = generate_city_bt_array(ev_count)
        print(f"✅ 成功构建车队物理模型：共计 {ev_count} 辆电动汽车。")
        
        # 2.3 分析自然负荷，构建电网峰谷上下文 (Grid Context)
        grid_ctx = build_grid_context(df_natural, load_col='Lnatural')
        print(f"✅ 成功构建电网感知：峰区 {grid_ctx['peak_hours']}，谷区 {grid_ctx['valley_hours']}")

        # 用于存储不同参与度下的调度负荷曲线与评估指标
        df_all_loads = pd.DataFrame({'Hour': range(24), 'Lnatural': natural_load})
        city_metrics_list = []
        
        start_city_time = time.time()
        
        # 2.4 遍历设定的用户参与度 (例如: 10% 到 100%)
        for pr in PARTICIPATION_RATES:
            pr_label = f"{pr*100:.0f}%"
            print(f"\n--- 启动参与度 {pr_label} 仿真，执行 {NUM_SIMULATION_RUNS} 次蒙特卡洛多进程聚合 ---")
            
            # 核心调度执行：调用多进程引擎获取净调度增量负荷
            net_dispatch_load = run_parallel_simulations(
                participation_rate=pr,
                grid_ctx=grid_ctx,
                bt_array=bt_array,
                total_runs=NUM_SIMULATION_RUNS,
                num_processes=NUM_PROCESSES,
                random_seed=RANDOM_SEED
            )
            
            # 公式(24): 计算综合调度电力负荷 (Ldispatch)
            dispatch_load = natural_load + net_dispatch_load
            
            # 将调度后的曲线存入汇总表
            df_all_loads[f'Ldispatch_{pr_label}'] = dispatch_load
            
            # 计算该参与度下的综合评估指标
            metrics = calculate_evaluation_metrics(natural_load, dispatch_load)
            metrics['City'] = city
            metrics['Participation_Rate'] = pr
            metrics['EV_Count'] = ev_count
            city_metrics_list.append(metrics)
            
            print(f"   [指标] 峰值削减率: {metrics['Peak_Reduction_Rate_%']:.2f}% | 负荷率: {metrics['Load_Ratio_Dispatch_%']:.2f}% | 峰谷差: {metrics['Delta_L_Dispatch']:.0f} kW")

        elapsed = time.time() - start_city_time
        print(f"\n🎉 城市 {city} 全参与度仿真完成，耗时 {elapsed:.2f} 秒。")

        # 2.5 导出当前城市的 24 小时负荷汇总曲线至 Excel
        formatted_city = city.replace(" ", "_")
        load_output_path = os.path.join(output_dir, f"{formatted_city}_V2G_Load_Curves.xlsx")
        
        # 导出当前城市的评估指标发展表
        df_metrics = pd.DataFrame(city_metrics_list)
        metrics_output_path = os.path.join(output_dir, f"{formatted_city}_V2G_Metrics.csv")
        
        try:
            df_all_loads.to_excel(load_output_path, index=False)
            df_metrics.to_csv(metrics_output_path, index=False)
            print(f"📄 城市 {city} 结果已保存至:\n  - {load_output_path}\n  - {metrics_output_path}")
        except Exception as e:
            print(f"[警告] 结果导出失败: {e}")
            
        all_cities_metrics.extend(city_metrics_list)

    # 3. 汇总所有城市的评估指标总表 (为最终的回归分析脚本准备数据)
    if all_cities_metrics:
        df_all_metrics = pd.DataFrame(all_cities_metrics)
        global_metrics_path = os.path.join(output_dir, "All_Cities_V2G_Metrics_Summary.csv")
        try:
            df_all_metrics.to_csv(global_metrics_path, index=False)
            print(f"\n✅ 跨区域所有城市综合评估指标已保存至: {global_metrics_path}")
        except Exception as e:
            print(f"[警告] 综合指标汇总导出失败: {e}")

    print("\n=== 04_run_v2g_simulation 执行完毕 ===")


if __name__ == "__main__":
    main()