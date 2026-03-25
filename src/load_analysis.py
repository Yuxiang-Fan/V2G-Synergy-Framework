import pandas as pd
import numpy as np
from typing import Tuple, Dict, List

def identify_peak_valley_hours(
    df_load: pd.DataFrame, 
    load_col: str = 'Lnatural', 
    time_col: str = 'Hour', 
    n_hours: int = 5
) -> Tuple[List[int], List[int], List[int]]:
    """基于自然负荷数据识别电网峰区、谷区与平段"""
    if len(df_load) != 24:
        raise ValueError(f"Expected 24 hours of load data, got {len(df_load)}")

    # 利用 DataFrame 内置方法快速提取极值索引
    peak_hours = df_load.nlargest(n_hours, load_col)[time_col].astype(int).tolist()
    valley_hours = df_load.nsmallest(n_hours, load_col)[time_col].astype(int).tolist()
    
    all_hours = set(range(24))
    flat_hours = list(all_hours - set(peak_hours) - set(valley_hours))
    
    return peak_hours, valley_hours, flat_hours

def calculate_charging_probabilities(
    df_load: pd.DataFrame, 
    peak_hours: List[int], 
    valley_hours: List[int], 
    load_col: str = 'Lnatural', 
    time_col: str = 'Hour'
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """计算峰区正向比例概率与谷区逆向加权概率"""
    df_indexed = df_load.set_index(time_col)
    
    # 峰区：负荷越大，放电权重越高 (正向比例)
    peak_loads = df_indexed.loc[peak_hours, load_col]
    peak_probs = (peak_loads / peak_loads.sum()).round(4).to_dict()
    
    # 谷区：负荷越小，充电权重越高 (逆向加权)
    valley_loads = df_indexed.loc[valley_hours, load_col]
    valley_inv_loads = 1.0 / valley_loads
    valley_probs = (valley_inv_loads / valley_inv_loads.sum()).round(4).to_dict()
    
    return peak_probs, valley_probs

def build_grid_context(df_load: pd.DataFrame, load_col: str = 'Lnatural') -> dict:
    """构建包含峰谷时段划分及抽样权重的全局 Context"""
    if 'hour' in df_load.columns and 'Hour' not in df_load.columns:
        df_load = df_load.rename(columns={'hour': 'Hour'})
        
    peak_hrs, valley_hrs, flat_hrs = identify_peak_valley_hours(df_load, load_col, time_col='Hour')
    peak_probs, valley_probs = calculate_charging_probabilities(df_load, peak_hrs, valley_hrs, load_col, time_col='Hour')
    
    return {
        "peak_hours": peak_hrs,
        "valley_hours": valley_hrs,
        "flat_hours": flat_hrs,
        "peak_sampling": {
            "hours": list(peak_probs.keys()),
            "probs": list(peak_probs.values())
        },
        "valley_sampling": {
            "hours": list(valley_probs.keys()),
            "probs": list(valley_probs.values())
        }
    }

if __name__ == "__main__":
    test_load_data = pd.DataFrame({
        'Hour': list(range(24)),
        'Lnatural': [2000, 1900, 1800, 1850, 2100, 2500, 3000, 4000, 4500, 4200, 
                     3800, 3500, 3600, 4000, 4800, 5500, 6000, 6500, 6800, 7000, 
                     6900, 6200, 5000, 3000]
    })
    
    grid_ctx = build_grid_context(test_load_data)
    
    print("Peak Hours:", grid_ctx['peak_hours'])
    print("Peak Probs:", grid_ctx['peak_sampling']['probs'])
    print("Valley Hours:", grid_ctx['valley_hours'])
    print("Valley Probs:", grid_ctx['valley_sampling']['probs'])
