import numpy as np
import pandas as pd
from scipy.stats import beta
import warnings

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

def sample_truncated_beta(bt_value: int, capacity: float, prev_power: float = None) -> float:
    """基于截断Beta分布的充电电量抽样，确保单次或多次充电时电量严格单调递增"""
    alpha, beta_param = BETA_PARAMS_MAP.get(bt_value, (1.0, 1.0))
    
    # 确定抽样下界：若存在前置电量则为前置电量，否则默认为容量的10%
    if prev_power is None:
        lower_bound = max(0.0, capacity * 0.1)
    else:
        lower_bound = float(prev_power)
        
    lower_bound = max(0.0, min(lower_bound, capacity))
    
    # 转换为 SOC 比例
    lower_ratio = lower_bound / float(capacity) if capacity > 0 else 0.0
    lower_ratio = min(max(lower_ratio, 0.0), 0.999999)
    
    # 在截断区间内进行均匀抽样并逆映射
    u = np.random.uniform(lower_ratio, 0.999999)
    sample_soc = beta.ppf(u, alpha, beta_param)
    
    if np.isnan(sample_soc):
        sample_soc = lower_ratio
        
    charge_amount = int(round(sample_soc * capacity))
    charge_amount = min(charge_amount, int(round(capacity)))
    
    if prev_power is not None:
        charge_amount = max(charge_amount, int(round(prev_power)))
        
    return float(charge_amount)


def initialize_ev_fleet(bt_array: np.ndarray, random_seed: int = None) -> pd.DataFrame:
    """初始化EV群体静态属性与初始SOC"""
    if random_seed is not None:
        np.random.seed(random_seed)
        
    df = pd.DataFrame({'BT': bt_array})
    df = df[df['BT'] != 0].copy()
    
    df['Battery_Capacity'] = df['BT'].map(BATTERY_CAPACITY_MAP).astype(float)
    df['Unit_Consumption'] = df['BT'].map(ENERGY_CONSUMPTION_MAP).astype(float)
    
    def sample_initial_soc(bt):
        a, b = BETA_PARAMS_MAP.get(bt, (1.0, 1.0))
        return float(np.random.beta(a, b))
        
    df['Initial_SOC'] = df['BT'].apply(sample_initial_soc)
    df['Initial_Power'] = (df['Battery_Capacity'] * df['Initial_SOC']).round().astype(float)
    
    return df


def simulate_daily_driving(df_ev: pd.DataFrame) -> pd.DataFrame:
    """模拟单日行驶耗电过程"""
    df = df_ev.copy()
    
    df['Mileage'] = np.random.choice(MILEAGE_BINS, size=len(df), p=MILEAGE_PROBS)
    df['Power_Consumption'] = (df['Mileage'] * df['Unit_Consumption']) / 100.0
    
    # 计算损耗后电量，兜底防止负值越界
    df['Remaining_Power'] = (df['Initial_Power'] - df['Power_Consumption']).clip(lower=0.0)
    df['Remaining_SOC'] = df['Remaining_Power'] / df['Battery_Capacity']
    
    return df


def simulate_base_charging_behavior(df_ev: pd.DataFrame) -> pd.DataFrame:
    """模拟无序基线充电行为"""
    df = df_ev.copy()
    
    # 抽样单日充电次数
    df['Charge_Count'] = np.random.choice(CHARGING_TIMES_BINS, size=len(df), p=CHARGING_TIMES_PROBS)
    
    def calculate_charging_stages(row):
        rem = float(row['Remaining_Power'])
        cap = float(row['Battery_Capacity'])
        count = int(row['Charge_Count'])
        bt = int(row['BT'])
        
        if count == 0:
            return pd.Series([rem, rem])
            
        elif count == 1:
            first_charge = sample_truncated_beta(bt, cap, prev_power=rem)
            return pd.Series([first_charge, first_charge])
            
        else:
            first_charge = sample_truncated_beta(bt, cap, prev_power=rem)
            second_charge = sample_truncated_beta(bt, cap, prev_power=first_charge)
            return pd.Series([first_charge, second_charge])
            
    df[['Stage_1_Power', 'Stage_2_Power']] = df.apply(calculate_charging_stages, axis=1, result_type='expand')
    df['Natural_Charge_Demand'] = (df['Stage_2_Power'] - df['Remaining_Power']).clip(lower=0.0)
    
    return df


def generate_ev_daily_profiles(bt_array: np.ndarray, random_seed: int = None) -> pd.DataFrame:
    """聚合EV物理状态的单日完整模拟流程"""
    df_init = initialize_ev_fleet(bt_array, random_seed)
    df_driven = simulate_daily_driving(df_init)
    
    return simulate_base_charging_behavior(df_driven)

if __name__ == "__main__":
    test_bt_array = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], size=1000, 
                                     p=[0.05, 0.1, 0.4, 0.1, 0.05, 0.15, 0.1, 0.05])
    
    df_profiles = generate_ev_daily_profiles(test_bt_array, random_seed=42)
    
    print("=== EV 初始状态与自然充电需求模拟 ===")
    summary = df_profiles[['Battery_Capacity', 'Initial_SOC', 'Mileage', 
                           'Remaining_SOC', 'Natural_Charge_Demand']].describe().round(2)
    print(summary)
    print(f"\n单日产生充电行为车辆占比: {(df_profiles['Charge_Count'] > 0).mean() * 100:.1f}%")
