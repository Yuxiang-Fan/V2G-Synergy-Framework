import os
import sys
import numpy as np
import pandas as pd
import multiprocessing

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
    """执行单次全城市车队的无序充电蒙特卡洛模拟"""
    df_ev = generate_ev_daily_profiles(bt_array, random_seed=seed)
    
    mask_charge = df_ev['Charge_Count'] > 0
    n_charging = mask_charge.sum()
    
    # 模拟无序充电的时间分布
    if n_charging > 0:
        df_ev.loc[mask_charge, 'Charge_Time'] = np.random.choice(
            range(24), size=n_charging, p=BASE_DEPARTURE_PROBS
        )
        
    # 聚合 24 小时负荷曲线
    load_curve = np.zeros(24)
    if n_charging > 0:
        valid_data = df_ev.loc[mask_charge, ['Charge_Time', 'Natural_Charge_Demand']].dropna()
        times = valid_data['Charge_Time'].astype(int).values
        amounts = valid_data['Natural_Charge_Demand'].values
        np.add.at(load_curve, times, amounts)
        
    return load_curve


def generate_city_bt_array(total_ev_count: int) -> np.ndarray:
    """基于洛杉矶调研抽样比例生成全城 EV 的 BT 物理分类数组"""
    base_ratios = np.array([0.004, 0.121, 0.568, 0.144, 0.018, 0.117, 0.003, 0.025])
    base_ratios = base_ratios / np.sum(base_ratios)
    
    counts = np.round(total_ev_count * base_ratios).astype(int)
    
    # 补齐四舍五入产生的尾数误差，归入占比最大的 BT3
    counts[2] += total_ev_count - counts.sum()  
    
    bt_list = []
    for bt_idx, cnt in enumerate(counts, start=1):
        bt_list.extend([bt_idx] * cnt)
        
    np.random.shuffle(bt_list)
    return np.array(bt_list)


def main():
    print("=== [03_run_baseline_load] 启动基础自然负荷提取 ===")
    
    output_dir = os.path.join(project_root, "results", "natural_loads")
    os.makedirs(output_dir, exist_ok=True)
    
    fallback_ev_counts = {
        "San Diego": 99183,
        "San Francisco": 28307,
        "Los Angeles": 282829,
        "New York": 189440,
        "Houston": 92519
    }
    
    for city in TARGET_CITIES:
        print(f"\n[*] 处理城市: {city}")
        
        try:
            df_synthesis = get_city_synthesis_load(city)
            synthesis_load = df_synthesis['Lsynthesis'].values
        except (FileNotFoundError, ValueError) as e:
            print(f"[Warning] 外部负荷数据加载失败，跳过 {city}: {e}")
            continue
            
        try:
            ev_count = get_city_ev_count(city, zip_codes=[]) 
        except FileNotFoundError:
            ev_count = fallback_ev_counts.get(city, 0)
            print(f"[Info] 启用调查统计常量，EV 保有量设定为: {ev_count}")
            
        if ev_count <= 0:
            print(f"[Skip] 城市 {city} EV 保有量异常。")
            continue
            
        bt_array = generate_city_bt_array(ev_count)
        
        print(f"[*] 启动多进程并行模拟 ({NUM_SIMULATION_RUNS} runs)...")
        processes = min(NUM_PROCESSES, multiprocessing.cpu_count(), NUM_SIMULATION_RUNS)
        
        tasks = []
        for i in range(NUM_SIMULATION_RUNS):
            tasks.append((1000 + i * 777, bt_array))
                    
        with multiprocessing.Pool(processes=processes) as pool:
            uncoordinated_loads = pool.starmap(worker_uncoordinated_charging, tasks)
            
        # 提取中位数作为基准无序充电分量
        median_ev_load = np.median(np.array(uncoordinated_loads), axis=0)
        
        # 差分计算纯净自然负荷 Lnatural，并执行防击穿限幅
        natural_load = np.maximum(synthesis_load - median_ev_load, 0.0)
        
        df_natural_out = pd.DataFrame({
            'Hour': range(24),
            'Lsynthesis': synthesis_load,
            'Median_EV_Charge': median_ev_load,
            'Lnatural': natural_load
        })
        
        out_path = os.path.join(output_dir, f"{city.replace(' ', '_')}.xlsx")
        
        try:
            df_natural_out.to_excel(out_path, index=False)
            print(f"[+] Lnatural 成功剥离并保存至: {out_path}")
            print(f"    日均 EV 负荷剥离量: {median_ev_load.mean():.2f} kW")
        except Exception as e:
            print(f"[Error] 保存 Excel 失败: {e}")

    print("\n=== [03_run_baseline_load] 执行完毕 ===")


if __name__ == "__main__":
    main()
