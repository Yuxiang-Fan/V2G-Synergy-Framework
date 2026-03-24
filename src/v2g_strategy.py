import os
import numpy as np
import pandas as pd
from scipy.stats import beta
import multiprocessing

from src.config import (
    BETA_PARAMS_MAP,
    BASE_DEPARTURE_PROBS,
    MILEAGE_BINS,
    MILEAGE_PROBS
)
from src.ev_simulation import generate_ev_daily_profiles


def sample_beta_bounded(bt_array: np.ndarray, lower_bound: float, upper_bound: float) -> np.ndarray:
    """基于逆变换采样的有界Beta分布向量化抽样"""
    samples = np.zeros(len(bt_array))
    unique_bts = np.unique(bt_array)
    
    for bt in unique_bts:
        mask = (bt_array == bt)
        a, b = BETA_PARAMS_MAP.get(bt, (1.0, 1.0))
        
        # 计算截断边界的CDF
        cdf_lower = beta.cdf(lower_bound, a, b)
        cdf_upper = beta.cdf(upper_bound, a, b)
        
        # 在截断区间内进行均匀抽样并逆映射
        u = np.random.uniform(cdf_lower, cdf_upper, size=mask.sum())
        samples[mask] = beta.ppf(u, a, b)
        
    return samples


def apply_v2g_strategy(df_base: pd.DataFrame, grid_ctx: dict, participation_rate: float) -> pd.DataFrame:
    """执行V2G时空负荷协同调度"""
    df = df_base.copy()
    n_evs = len(df)
    
    # 确定V2G参与意愿
    df['Is_Participating'] = (np.random.rand(n_evs) <= participation_rate).astype(int)
    
    if 'Departure_Hour' not in df.columns:
        df['Departure_Hour'] = np.random.choice(range(24), size=n_evs, p=BASE_DEPARTURE_PROBS)
    
    # 用户群分类映射: 0-未参与, 1-A类(峰区), 2-B类(谷区), 3-C类(平段)
    conditions = [
        df['Is_Participating'] == 0,
        df['Departure_Hour'].isin(grid_ctx['peak_hours']),
        df['Departure_Hour'].isin(grid_ctx['valley_hours'])
    ]
    df['User_Type'] = np.select(conditions, [0, 1, 2], default=3)
    
    # 初始化输出列
    df['Final_Charge_Demand'] = df.get('Natural_Charge_Demand', 0.0)
    df['Scheduled_Charge_Time'] = np.nan
    
    # 1. 非参与者与B类用户: 维持自然无序充电
    mask_natural = df['User_Type'].isin([0, 2])
    df.loc[mask_natural, 'Scheduled_Charge_Time'] = np.random.choice(
        range(24), size=mask_natural.sum(), p=BASE_DEPARTURE_PROBS
    )
    
    # 2. A类用户: 负荷平移至谷区
    mask_a = df['User_Type'] == 1
    if mask_a.sum() > 0:
        df.loc[mask_a, 'Scheduled_Charge_Time'] = np.random.choice(
            grid_ctx['valley_sampling']['hours'], 
            size=mask_a.sum(), 
            p=grid_ctx['valley_sampling']['probs']
        )
        
    # 3. C类用户: V2G双向充放电调度
    mask_c = df['User_Type'] == 3
    if mask_c.sum() > 0:
        bt_c = df.loc[mask_c, 'BT'].values
        cap_c = df.loc[mask_c, 'Battery_Capacity'].values
        unit_c = df.loc[mask_c, 'Unit_Consumption'].values
        
        # 初始高SOC与耗电量估算
        soc_init = sample_beta_bounded(bt_c, 0.9, 0.9999)
        power_init = cap_c * soc_init
        
        mileage = np.random.choice(MILEAGE_BINS, size=mask_c.sum(), p=MILEAGE_PROBS)
        power_consume = (mileage * unit_c) / 100.0
        power_remain = np.maximum(power_init - power_consume, 0.0)
        
        # 设定V2G放电下限并筛选可行车辆
        soc_target = sample_beta_bounded(bt_c, 0.2, 0.3)
        power_target = cap_c * soc_target
        active_v2g = power_remain > power_target
        
        # 峰区放电 (V2G)
        df.loc[mask_c, 'V2G_Discharge_Amount'] = np.where(active_v2g, power_remain - power_target, 0.0)
        df.loc[mask_c, 'V2G_Discharge_Time'] = np.random.choice(
            grid_ctx['peak_sampling']['hours'], 
            size=mask_c.sum(), 
            p=grid_ctx['peak_sampling']['probs']
        )
        
        # 谷区充电 (G2V) 补齐电量
        df.loc[mask_c, 'V2G_Charge_Amount'] = np.where(active_v2g, power_init - power_target, 0.0)
        df.loc[mask_c, 'V2G_Charge_Time'] = np.random.choice(
            grid_ctx['valley_sampling']['hours'], 
            size=mask_c.sum(), 
            p=grid_ctx['valley_sampling']['probs']
        )

    return df


def aggregate_load_arrays(df: pd.DataFrame) -> dict:
    """按小时维度聚合不同调度类型的负荷"""
    
    def _accumulate(arr, times, amounts):
        """处理时间映射与就地累加"""
        valid = ~np.isnan(times)
        if valid.any():
            np.add.at(arr, times[valid].astype(int), amounts[valid].values)

    loads = {
        'a': np.zeros(24),
        'b0': np.zeros(24),
        'c_charge': np.zeros(24),
        'c_discharge': np.zeros(24)
    }

    mask_a = df['User_Type'] == 1
    _accumulate(loads['a'], df.loc[mask_a, 'Scheduled_Charge_Time'].values, df.loc[mask_a, 'Final_Charge_Demand'])
    
    mask_b0 = df['User_Type'].isin([0, 2])
    _accumulate(loads['b0'], df.loc[mask_b0, 'Scheduled_Charge_Time'].values, df.loc[mask_b0, 'Final_Charge_Demand'])
    
    if 'V2G_Charge_Time' in df.columns:
        _accumulate(loads['c_charge'], df['V2G_Charge_Time'].values, df['V2G_Charge_Amount'])
        _accumulate(loads['c_discharge'], df['V2G_Discharge_Time'].values, df['V2G_Discharge_Amount'])
        
    return loads


def worker_simulation(seed: int, participation_rate: float, grid_ctx: dict, num_runs: int, bt_array: np.ndarray) -> dict:
    """子进程任务：执行多次单日蒙特卡洛仿真并返回累加负荷"""
    sums = {k: np.zeros(24) for k in ['a', 'b0', 'c_charge', 'c_discharge']}
    
    for i in range(num_runs):
        sim_seed = None if seed is None else seed + i
        
        df_base = generate_ev_daily_profiles(bt_array, random_seed=sim_seed)
        df_base['Departure_Hour'] = np.random.choice(range(24), size=len(df_base), p=BASE_DEPARTURE_PROBS)
        
        df_v2g = apply_v2g_strategy(df_base, grid_ctx, participation_rate)
        res = aggregate_load_arrays(df_v2g)
        
        for k in sums.keys():
            sums[k] += res[k]

    sums['count'] = num_runs
    return sums


def run_parallel_simulations(participation_rate: float, grid_ctx: dict, bt_array: np.ndarray, 
                             total_runs: int = 1000, num_processes: int = 64, random_seed: int = 12345) -> np.ndarray:
    """分配并行任务池，获取全局24小时净调度负荷期望"""
    processes = min(num_processes, multiprocessing.cpu_count(), total_runs)
    runs_per_proc = total_runs // processes
    remainder = total_runs % processes
    
    tasks = []
    for i in range(processes):
        runs_i = runs_per_proc + (1 if i < remainder else 0)
        if runs_i > 0:
            seed_i = None if random_seed is None else (random_seed + i * 100000)
            tasks.append((seed_i, participation_rate, grid_ctx, runs_i, bt_array))
            
    with multiprocessing.Pool(processes) as pool:
        results = pool.starmap(worker_simulation, tasks)
        
    total_count = sum(res['count'] for res in results)
    if total_count == 0:
        raise ValueError("Simulation iteration count is 0.")
        
    # 计算各项负荷均值
    avg = {k: sum(res[k] for res in results) / total_count for k in ['a', 'b0', 'c_charge', 'c_discharge']}
    
    # 净负荷 = 自然充电 + 谷区转移充电 + V2G充电 - V2G放电
    net_dispatch_load = (avg['b0'] + avg['a'] + avg['c_charge']) - avg['c_discharge']
    
    return net_dispatch_load
