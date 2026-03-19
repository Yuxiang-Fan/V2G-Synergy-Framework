"""
执行脚本 03: 提取城市基础自然负荷 (03_run_baseline_load.py)
执行差分算法，通过蒙特卡洛模拟剥离无序充电分量，提取出纯净的城市基础电力负荷 (Lnatural)。
为后续的有序充电策略 (V2G) 评估提供关键的对比基准。
"""

import os
import sys
import numpy as np
import pandas as pd
import multiprocessing

# 动态挂载项目根目录到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import (
    TARGET_CITIES, 
    NUM_SIMULATION_RUNS, 
    NUM_PROCESSES,
    BASE_DEPARTURE_PROBS
)
from src.data_loader import get_city_synthesis_load, get_city_ev_count
from src.ev_simulation import generate_ev_daily_profiles


def worker_uncoordinated_charging(seed: int, bt_array: np.ndarray) -> np.ndarray:
    """
    多进程 Worker：执行单次全城市车队的无序充电模拟。
    
    :param seed: 随机种子
    :param bt_array: 城市所有电动汽车的 BT 分类数组
    :return: 长度为 24 的单日无序充电总负荷数组
    """
    # 1. 生成单日耗电与自然充电需求 (基础物理状态)
    df_ev = generate_ev_daily_profiles(bt_array, random_seed=seed)
    
    # 2. 模拟无序充电的时间分布 (按照基础出行时间/充电概率抽取)
    mask_charge = df_ev['Charge_Count'] > 0
    n_charging = mask_charge.sum()
    
    if n_charging > 0:
        charge_times = np.random.choice(range(24), size=n_charging, p=BASE_DEPARTURE_PROBS)
        df_ev.loc[mask_charge, 'Charge_Time'] = charge_times
        
    # 3. 聚合成 24 小时负荷曲线
    load_curve = np.zeros(24)
    if n_charging > 0:
        valid_data = df_ev.loc[mask_charge, ['Charge_Time', 'Natural_Charge_Demand']].dropna()
        times = valid_data['Charge_Time'].astype(int).values
        amounts = valid_data['Natural_Charge_Demand'].values
        np.add.at(load_curve, times, amounts)
        
    return load_curve


def generate_city_bt_array(total_ev_count: int) -> np.ndarray:
    """
    根据给定的总保有量，按照典型比例分配生成车辆 BT 等级数组。
    (默认比例参考自洛杉矶的调研抽样数据 [BT1~8])
    """
    # 典型抽样比例分布 (BT1 到 BT8)
    base_ratios = [0.004, 0.121, 0.568, 0.144, 0.018, 0.117, 0.003, 0.025]
    
    # 确保比例和为 1
    base_ratios = np.array(base_ratios) / np.sum(base_ratios)
    
    # 生成各分类的具体数量
    counts = np.round(total_ev_count * base_ratios).astype(int)
    
    # 微调以确保总数绝对一致
    diff = total_ev_count - counts.sum()
    counts[2] += diff  # 把舍入误差补到占比最大的 BT3 上
    
    bt_list = []
    for bt_idx, cnt in enumerate(counts, start=1):
        bt_list.extend([bt_idx] * cnt)
        
    np.random.shuffle(bt_list)
    return np.array(bt_list)


def main():
    print("=== 开始执行: 03_run_baseline_load ===")
    
    # 1. 创建输出目录
    output_dir = os.path.join(project_root, "results", "natural_loads")
    os.makedirs(output_dir, exist_ok=True)
    
    # 备用字典：如果外部 EV 数据集未配置，使用论文表1中的统计常量作为 Fallback
    fallback_ev_counts = {
        "San Diego": 99183,
        "San Francisco": 28307,
        "Los Angeles": 282829,
        "New York": 189440,
        "Houston": 92519
    }
    
    # 2. 遍历目标城市，提取基础负荷
    for city in TARGET_CITIES:
        print(f"\n--- 正在处理城市: {city} ---")
        
        # 2.1 获取综合电力负荷 (Lsynthesis)
        try:
            df_synthesis = get_city_synthesis_load(city)
            synthesis_load = df_synthesis['Lsynthesis'].values
        except FileNotFoundError:
            print(f"[警告] 外部综合负荷数据未找到，跳过城市 {city}。")
            print("请确保 EIA 负荷数据存在于 config.py 指定的路径中。")
            continue
        except ValueError as e:
            print(f"[警告] {e} 跳过。")
            continue
            
        # 2.2 获取该城市的电动汽车保有量
        try:
            # 此处的 zip_codes 列表需使用者在真实环境下传入，这里传入空列表以触发Fallback
            ev_count = get_city_ev_count(city, zip_codes=[]) 
        except FileNotFoundError:
            ev_count = fallback_ev_counts.get(city, 0)
            print(f"未能读取外部 EV 注册表，采用论文调查统计常量: {ev_count} 辆。")
            
        if ev_count <= 0:
            print(f"[跳过] 城市 {city} 的 EV 保有量异常 ({ev_count})。")
            continue
            
        # 2.3 生成城市电动汽车 BT 物理分类数组
        bt_array = generate_city_bt_array(ev_count)
        
        # 2.4 并行执行蒙特卡洛无序充电模拟 (剥离基线分量)
        print(f"启动多进程蒙特卡洛模拟 ({NUM_SIMULATION_RUNS} 次)，计算百万级车队的自然充电负荷...")
        
        cpu_cnt = multiprocessing.cpu_count()
        processes = min(NUM_PROCESSES, cpu_cnt, NUM_SIMULATION_RUNS)
        runs_per_proc = NUM_SIMULATION_RUNS // processes
        remainder = NUM_SIMULATION_RUNS % processes
        
        tasks = []
        for i in range(processes):
            r_i = runs_per_proc + (1 if i < remainder else 0)
            if r_i > 0:
                seed_i = 1000 + i * 777
                for _ in range(r_i):
                    # 为了兼容 map，这里展开了单次运行任务
                    tasks.append((seed_i + _, bt_array))
                    
        with multiprocessing.Pool(processes=processes) as pool:
            uncoordinated_loads = pool.starmap(worker_uncoordinated_charging, tasks)
            
        # 2.5 差分算法计算：采取中位数聚法 (Median) 提取基准充电分量
        uncoordinated_loads_matrix = np.array(uncoordinated_loads) # shape: (runs, 24)
        median_ev_load = np.median(uncoordinated_loads_matrix, axis=0)
        
        # L_natural = L_synthesis - Median_EV_Load
        natural_load = synthesis_load - median_ev_load
        
        # 物理兜底机制：自然负荷不能低于0 (防止极端缩放错误)
        natural_load = np.maximum(natural_load, 0.0)
        
        # 2.6 整理导出
        df_natural_out = pd.DataFrame({
            'Hour': range(24),
            'Lsynthesis': synthesis_load,
            'Median_EV_Charge': median_ev_load,
            'Lnatural': natural_load
        })
        
        formatted_city = city.replace(" ", "_")
        out_path = os.path.join(output_dir, f"{formatted_city}.xlsx")
        
        try:
            df_natural_out.to_excel(out_path, index=False)
            print(f"✅ {city} 的自然电力负荷基线已成功提取并保存至: {out_path}")
            print(f"   [校验] 日均剥离 EV 负荷: {median_ev_load.mean():.2f} kW")
        except Exception as e:
            print(f"[错误] 保存 Excel 失败: {e}")

    print("\n=== 03_run_baseline_load 执行完毕 ===")


if __name__ == "__main__":
    main()