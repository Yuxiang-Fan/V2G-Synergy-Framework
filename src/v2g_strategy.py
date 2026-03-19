"""
电动汽车有序充电策略与多进程调度模块 (v2g_strategy.py)
实现 A/B/C 类用户的分类 V2G 协同调度，并基于 Multiprocessing 提供大规模并行仿真框架。
"""

import os
import time
import numpy as np
import pandas as pd
from scipy.stats import beta
import multiprocessing

# 引入全局配置
from src.config import (
    BETA_PARAMS_MAP,
    BASE_DEPARTURE_PROBS,
    MILEAGE_BINS,
    MILEAGE_PROBS
)

# 引入基础物理模型
from src.ev_simulation import generate_ev_daily_profiles


def sample_beta_bounded(bt_array: np.ndarray, lower_bound: float, upper_bound: float) -> np.ndarray:
    """
    向量化的高效有界 Beta 分布抽样函数 (利用逆 CDF 变换法)。
    比原来 python 里的 for 循环重抽样快 100 倍以上。
    
    :param bt_array: 车辆分类等级数组 (用于获取 alpha, beta 参数)
    :param lower_bound: 抽样下限
    :param upper_bound: 抽样上限
    :return: 处于 [lower_bound, upper_bound] 之间的 SOC 数组
    """
    samples = np.zeros(len(bt_array))
    
    # 获取唯一的 BT 分类，批量处理以加速
    unique_bts = np.unique(bt_array)
    for bt in unique_bts:
        mask = (bt_array == bt)
        a, b = BETA_PARAMS_MAP.get(bt, (1.0, 1.0))
        
        # 计算边界在当前 Beta 分布下的 CDF 值
        cdf_lower = beta.cdf(lower_bound, a, b)
        cdf_upper = beta.cdf(upper_bound, a, b)
        
        # 在 CDF 边界内进行均匀抽样
        u = np.random.uniform(cdf_lower, cdf_upper, size=mask.sum())
        
        # 逆运算得到有界的真实 SOC
        samples[mask] = beta.ppf(u, a, b)
        
    return samples


def apply_v2g_strategy(df_base: pd.DataFrame, grid_ctx: dict, participation_rate: float) -> pd.DataFrame:
    """
    执行单次有序充电调度策略，生成各类型用户的时空负荷响应。
    严格依据论文划分 A类(峰区)、B类(谷区)、C类(平段) 并应用相应的充放电逻辑。
    
    :param df_base: 由 ev_simulation.py 生成的含有自然耗电与充电需求的 DataFrame
    :param grid_ctx: 由 load_analysis.py 生成的电网峰谷上下文
    :param participation_rate: 当前模拟的用户参与度 (0.0 ~ 1.0)
    :return: 附加了 V2G 调度负荷与时间标签的 DataFrame
    """
    df = df_base.copy()
    n_evs = len(df)
    
    # 1. 生成参与标志
    df['Is_Participating'] = (np.random.rand(n_evs) <= participation_rate).astype(int)
    
    # 提取时间与上下文参数
    dep_hours = df['Departure_Hour'].values if 'Departure_Hour' in df.columns else np.random.choice(range(24), size=n_evs, p=BASE_DEPARTURE_PROBS)
    df['Departure_Hour'] = dep_hours
    
    peak_hours = grid_ctx['peak_hours']
    valley_hours = grid_ctx['valley_hours']
    
    # 2. 判定峰谷区 (A/B/C 类用户分类)
    # 0: 未参与, 1: A类(峰区), 2: B类(谷区), 3: C类(平段)
    conditions = [
        df['Is_Participating'] == 0,
        df['Departure_Hour'].isin(peak_hours),
        df['Departure_Hour'].isin(valley_hours)
    ]
    choices = [0, 1, 2]
    df['User_Type'] = np.select(conditions, choices, default=3)
    
    # ==========================================
    # 3. 处理非参与者与 B类用户 (自然充电)
    # ==========================================
    mask_natural = df['User_Type'].isin([0, 2])
    df.loc[mask_natural, 'Final_Charge_Demand'] = df.loc[mask_natural, 'Natural_Charge_Demand']
    # 自然充电时间按基础概率重新分布 (模拟无序性)
    df.loc[mask_natural, 'Scheduled_Charge_Time'] = np.random.choice(range(24), size=mask_natural.sum(), p=BASE_DEPARTURE_PROBS)
    
    # ==========================================
    # 4. 处理 A类用户 (强制谷区充电)
    # ==========================================
    mask_a = df['User_Type'] == 1
    df.loc[mask_a, 'Final_Charge_Demand'] = df.loc[mask_a, 'Natural_Charge_Demand']
    # 依据论文：将其充电时间转移至谷区，按逆向加权概率分布
    if mask_a.sum() > 0:
        df.loc[mask_a, 'Scheduled_Charge_Time'] = np.random.choice(
            grid_ctx['valley_sampling']['hours'], 
            size=mask_a.sum(), 
            p=grid_ctx['valley_sampling']['probs']
        )
        
    # ==========================================
    # 5. 处理 C类用户 (V2G 双向充电)
    # ==========================================
    mask_c = df['User_Type'] == 3
    if mask_c.sum() > 0:
        bt_c = df.loc[mask_c, 'BT'].values
        cap_c = df.loc[mask_c, 'Battery_Capacity'].values
        unit_c = df.loc[mask_c, 'Unit_Consumption'].values
        
        # 初始电量设定：抽取 >= 0.9 的高 SOC
        soc_init_c = sample_beta_bounded(bt_c, 0.9, 0.9999)
        power_init_c = cap_c * soc_init_c
        
        # 耗电量计算 (抽样里程)
        mileage_c = np.random.choice(MILEAGE_BINS, size=mask_c.sum(), p=MILEAGE_PROBS)
        power_consume_c = (mileage_c * unit_c) / 100.0
        power_remain_c = np.maximum(power_init_c - power_consume_c, 0.0)
        
        # 目标反向放电设定：保留 20% ~ 30% SOC
        soc_target_c = sample_beta_bounded(bt_c, 0.2, 0.3)
        power_target_c = cap_c * soc_target_c
        
        # 判定是否具备放电条件：剩余电量 > 目标下限电量
        active_v2g_mask = power_remain_c > power_target_c
        
        # --- 反向放电 (Discharge to grid, 削峰) ---
        discharge_amount = np.where(active_v2g_mask, power_remain_c - power_target_c, 0.0)
        df.loc[mask_c, 'V2G_Discharge_Amount'] = discharge_amount
        # 放电时间：发生在负荷峰区，按峰区比例权重分布
        df.loc[mask_c, 'V2G_Discharge_Time'] = np.random.choice(
            grid_ctx['peak_sampling']['hours'], 
            size=mask_c.sum(), 
            p=grid_ctx['peak_sampling']['probs']
        )
        
        # --- 正向充电 (Charge from grid, 填谷) ---
        # 充回初始高电量状态 (或者也可以再抽样一次 target >= 0.9)
        charge_amount = np.where(active_v2g_mask, power_init_c - power_target_c, 0.0)
        df.loc[mask_c, 'V2G_Charge_Amount'] = charge_amount
        # 充电时间：发生在负荷谷区，按谷区逆向加权分布
        df.loc[mask_c, 'V2G_Charge_Time'] = np.random.choice(
            grid_ctx['valley_sampling']['hours'], 
            size=mask_c.sum(), 
            p=grid_ctx['valley_sampling']['probs']
        )

    return df


def aggregate_load_arrays(df: pd.DataFrame) -> dict:
    """
    将用户的离散充放电事件按小时聚合成 24 维度的 NumPy 数组。
    """
    arr_natural_a = np.zeros(24)
    arr_natural_b0 = np.zeros(24)
    arr_v2g_charge = np.zeros(24)
    arr_v2g_discharge = np.zeros(24)
    
    def add_to_array(arr, times, amounts):
        valid = ~np.isnan(times)
        if valid.sum() == 0: return
        t_valid = times[valid].astype(int)
        a_valid = amounts[valid].values
        np.add.at(arr, t_valid, a_valid)

    # 聚合 A类 调度后充电量
    mask_a = df['User_Type'] == 1
    add_to_array(arr_natural_a, df.loc[mask_a, 'Scheduled_Charge_Time'].values, df.loc[mask_a, 'Final_Charge_Demand'])
    
    # 聚合 非参与及B类 自然充电量
    mask_b0 = df['User_Type'].isin([0, 2])
    add_to_array(arr_natural_b0, df.loc[mask_b0, 'Scheduled_Charge_Time'].values, df.loc[mask_b0, 'Final_Charge_Demand'])
    
    # 聚合 C类 V2G 充放电量
    if 'V2G_Charge_Time' in df.columns:
        add_to_array(arr_v2g_charge, df['V2G_Charge_Time'].values, df['V2G_Charge_Amount'])
        add_to_array(arr_v2g_discharge, df['V2G_Discharge_Time'].values, df['V2G_Discharge_Amount'])
        
    return {
        'arr_a': arr_natural_a,
        'arr_b0': arr_natural_b0,
        'arr_c_charge': arr_v2g_charge,
        'arr_c_discharge': arr_v2g_discharge
    }


def worker_simulation(seed: int, participation_rate: float, grid_ctx: dict, num_runs: int, bt_array: np.ndarray) -> dict:
    """
    多进程 Worker 函数：执行 num_runs 次完整单日仿真并累加结果。
    """
    pid = os.getpid()
    
    sum_a = np.zeros(24)
    sum_b0 = np.zeros(24)
    sum_c_charge = np.zeros(24)
    sum_c_discharge = np.zeros(24)
    
    for i in range(num_runs):
        sim_seed = None if seed is None else seed + i
        
        # 1. 生成基线数据 (调用 ev_simulation)
        df_base = generate_ev_daily_profiles(bt_array, random_seed=sim_seed)
        # 为基线赋予出行时间特征
        df_base['Departure_Hour'] = np.random.choice(range(24), size=len(df_base), p=BASE_DEPARTURE_PROBS)
        
        # 2. 执行 V2G 调度策略
        df_v2g = apply_v2g_strategy(df_base, grid_ctx, participation_rate)
        
        # 3. 结果聚合
        res = aggregate_load_arrays(df_v2g)
        sum_a += res['arr_a']
        sum_b0 += res['arr_b0']
        sum_c_charge += res['arr_c_charge']
        sum_c_discharge += res['arr_c_discharge']

    return {
        'sum_a': sum_a,
        'sum_b0': sum_b0,
        'sum_c_charge': sum_c_charge,
        'sum_c_discharge': sum_c_discharge,
        'count': num_runs
    }


def run_parallel_simulations(participation_rate: float, grid_ctx: dict, bt_array: np.ndarray, 
                             total_runs: int = 1000, num_processes: int = 64, random_seed: int = 12345) -> np.ndarray:
    """
    管理并行进程池，执行指定参与度下的全面仿真，并返回 24 小时净调度负荷。
    """
    cpu_cnt = multiprocessing.cpu_count()
    processes = min(num_processes, cpu_cnt, total_runs)
    
    runs_per_process = total_runs // processes
    remainder = total_runs % processes
    
    tasks = []
    for i in range(processes):
        runs_i = runs_per_process + (1 if i < remainder else 0)
        if runs_i > 0:
            seed_i = None if random_seed is None else (random_seed + i * 100000)
            tasks.append((seed_i, participation_rate, grid_ctx, runs_i, bt_array))
            
    with multiprocessing.Pool(processes=len(tasks)) as pool:
        results = pool.starmap(worker_simulation, tasks)
        
    # 汇总所有 Worker 结果
    total_a = np.zeros(24)
    total_b0 = np.zeros(24)
    total_c_charge = np.zeros(24)
    total_c_discharge = np.zeros(24)
    total_count = 0
    
    for res in results:
        total_a += res['sum_a']
        total_b0 += res['sum_b0']
        total_c_charge += res['sum_c_charge']
        total_c_discharge += res['sum_c_discharge']
        total_count += res['count']
        
    if total_count == 0:
        raise ValueError("模拟次数分配异常，总执行次数为 0。")
        
    # 计算均值
    avg_a = total_a / total_count
    avg_b0 = total_b0 / total_count
    avg_c_charge = total_c_charge / total_count
    avg_c_discharge = total_c_discharge / total_count
    
    # 计算调度净增量 = (自然充电 + 强制谷区充电 + C类正向充电) - C类反向放电
    net_dispatch_load = (avg_b0 + avg_a + avg_c_charge) - avg_c_discharge
    
    return net_dispatch_load