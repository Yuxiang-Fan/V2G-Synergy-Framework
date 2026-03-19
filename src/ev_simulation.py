"""
电动汽车单日行为模拟模块 (ev_simulation.py)
负责电动汽车群体的静态参数初始化、动态行驶里程抽样、单日耗电量计算，
以及多阶段自然充电行为（Beta分布逆变换抽样）的模拟。
"""

import numpy as np
import pandas as pd
from scipy.stats import beta
import warnings

# 引入全局配置参数
from src.config import (
    BATTERY_CAPACITY_MAP,
    ENERGY_CONSUMPTION_MAP,
    BETA_PARAMS_MAP,
    CHARGING_TIMES_BINS,
    CHARGING_TIMES_PROBS,
    MILEAGE_BINS,
    MILEAGE_PROBS
)

warnings.filterwarnings('ignore', category=RuntimeWarning)

def calculate_charge_optimized(bt_value: int, current_capacity: float, previous_value: float = None) -> float:
    """
    基于 Beta 分布的逆 CDF 法，计算电动汽车的充电后电量。
    在循环充电时，严格满足递增性条件 (公式 8: SOC_k > SOC_{k-1})。
    
    :param bt_value: 车辆分类等级 (1-8)
    :param current_capacity: 电池总容量 (kWh)
    :param previous_value: 上一次充电后的电量或损耗后剩余电量 (kWh)
    :return: 充电后的电量绝对值 (kWh)
    """
    # 获取该类车型的 Beta 分布形状参数
    alpha, beta_param = BETA_PARAMS_MAP.get(bt_value, (1.0, 1.0))
    
    # 确定抽样的下限边界
    if previous_value is None:
        # 如果是首次充电且无前置状态，默认至少充到 10%
        lower_bound = max(0.0, current_capacity * 0.1)
    else:
        # 如果存在前置状态，必须大于等于前置电量
        lower_bound = float(previous_value)
        
    lower_bound = max(0.0, min(lower_bound, current_capacity))
    
    # 将电量下限转化为 SOC 比例下限
    lower_ratio = lower_bound / float(current_capacity) if current_capacity > 0 else 0.0
    lower_ratio = min(max(lower_ratio, 0.0), 0.999999)
    
    # 利用均匀分布条件抽样 U ~ U(lower_ratio, 1.0)，并通过逆 CDF 获取目标 SOC
    u = np.random.uniform(lower_ratio, 0.999999)
    sample_soc = beta.ppf(u, alpha, beta_param)
    
    if np.isnan(sample_soc):
        sample_soc = lower_ratio
        
    # 计算实际充电量，并将其限制在物理容量边界内
    charge_amount = int(round(sample_soc * current_capacity))
    charge_amount = min(charge_amount, int(round(current_capacity)))
    
    if previous_value is not None:
        charge_amount = max(charge_amount, int(round(previous_value)))
        
    return float(charge_amount)


def initialize_ev_fleet(bt_array: np.ndarray, random_seed: int = None) -> pd.DataFrame:
    """
    初始化电动汽车群体静态物理参量与初始荷电状态。
    
    :param bt_array: 包含车辆分类等级(1-8)的数组
    :param random_seed: 随机种子，确保可复现性
    :return: 包含初始化特征的 DataFrame
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    df = pd.DataFrame({'BT': bt_array})
    df = df[df['BT'] != 0].copy()
    
    # 映射电池容量与单位能耗
    df['Battery_Capacity'] = df['BT'].map(BATTERY_CAPACITY_MAP).astype(float)
    df['Unit_Consumption'] = df['BT'].map(ENERGY_CONSUMPTION_MAP).astype(float)
    
    # 抽样生成初始 SOC (Beta 分布)
    def sample_initial_soc(bt):
        a, b = BETA_PARAMS_MAP.get(bt, (1.0, 1.0))
        return float(np.random.beta(a, b))
        
    df['Initial_SOC'] = df['BT'].apply(sample_initial_soc)
    df['Initial_Power'] = (df['Battery_Capacity'] * df['Initial_SOC']).round().astype(float)
    
    return df


def simulate_daily_driving(df_ev: pd.DataFrame) -> pd.DataFrame:
    """
    模拟电动汽车的单日行驶耗电过程 (公式 2 与 公式 3)。
    
    :param df_ev: 初始化后的电动汽车 DataFrame
    :return: 包含里程、耗电量、剩余电量的 DataFrame
    """
    df = df_ev.copy()
    
    # 按概率抽样行驶里程
    df['Mileage'] = np.random.choice(MILEAGE_BINS, size=len(df), p=MILEAGE_PROBS)
    
    # 公式(2): 计算单日耗电量 (E_consume = e * S / 100)
    df['Power_Consumption'] = (df['Mileage'] * df['Unit_Consumption']) / 100.0
    
    # 公式(3): 计算损耗后电量与剩余 SOC (防击穿处理，最低为0)
    df['Remaining_Power'] = (df['Initial_Power'] - df['Power_Consumption']).clip(lower=0.0)
    df['Remaining_SOC'] = df['Remaining_Power'] / df['Battery_Capacity']
    
    return df


def simulate_base_charging_behavior(df_ev: pd.DataFrame) -> pd.DataFrame:
    """
    模拟电动汽车的自然多阶段充电行为 (不含 V2G 有序调度约束)。
    对应论文中的泊松分布充电次数及 Beta 分布递归充电量累加。
    
    :param df_ev: 包含剩余电量信息的 DataFrame
    :return: 附加了各阶段充电量及总充电需求的 DataFrame
    """
    df = df_ev.copy()
    
    # 抽样充电次数 (0, 1, 或 2次)
    df['Charge_Count'] = np.random.choice(CHARGING_TIMES_BINS, size=len(df), p=CHARGING_TIMES_PROBS)
    
    def calculate_charging_stages(row):
        rem = float(row['Remaining_Power'])
        cap = float(row['Battery_Capacity'])
        count = int(row['Charge_Count'])
        bt = int(row['BT'])
        
        if count == 0:
            # 不充电，状态保持
            first_charge = rem
            second_charge = rem
            
        elif count == 1:
            # 仅充电一次
            first_charge = calculate_charge_optimized(bt, cap, previous_value=rem)
            second_charge = first_charge
            
        else:
            # 充电两次，需满足递增性约束
            first_charge = calculate_charge_optimized(bt, cap, previous_value=rem)
            second_charge = calculate_charge_optimized(bt, cap, previous_value=first_charge)
            
        return pd.Series([first_charge, second_charge])
        
    # 应用多阶段充电计算
    df[['Stage_1_Power', 'Stage_2_Power']] = df.apply(calculate_charging_stages, axis=1, result_type='expand')
    
    # 计算自然充电模式下的总充电需求量 (公式 10)
    df['Natural_Charge_Demand'] = (df['Stage_2_Power'] - df['Remaining_Power']).clip(lower=0.0)
    
    return df


def generate_ev_daily_profiles(bt_array: np.ndarray, random_seed: int = None) -> pd.DataFrame:
    """
    全流程执行函数：整合初始化、耗电与基础充电行为。
    供基线负荷分析或后续 V2G 策略模块调用。
    
    :param bt_array: 车辆分类等级数组 (代表整个城市的电动汽车分布)
    :param random_seed: 随机种子
    :return: 包含完整单日物理状态的 DataFrame
    """
    # 1. 物理参数初始化
    df_init = initialize_ev_fleet(bt_array, random_seed)
    
    # 2. 行驶与耗电模拟
    df_driven = simulate_daily_driving(df_init)
    
    # 3. 自然充电行为模拟
    df_final = simulate_base_charging_behavior(df_driven)
    
    return df_final

# ==========================================
# 测试代码 (模块内独立运行时执行)
# ==========================================
if __name__ == "__main__":
    # 模拟洛杉矶的一个微缩车队 (1000辆车，BT类型分布)
    test_bt_array = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], size=1000, 
                                     p=[0.05, 0.1, 0.4, 0.1, 0.05, 0.15, 0.1, 0.05])
    
    print("正在模拟 1000 辆 EV 的单日耗电与自然充电过程...")
    df_profiles = generate_ev_daily_profiles(test_bt_array, random_seed=42)
    
    print("\n【模拟结果摘要】")
    summary = df_profiles[['Battery_Capacity', 'Initial_SOC', 'Mileage', 
                           'Remaining_SOC', 'Charge_Count', 'Natural_Charge_Demand']].describe().round(2)
    print(summary)
    
    print("\n单日有充电行为的车辆比例: {:.1f}%".format(
        (df_profiles['Charge_Count'] > 0).mean() * 100
    ))