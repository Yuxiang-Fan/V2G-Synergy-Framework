"""
电网负荷分析模块 (load_analysis.py)
负责根据城市的自然电力负荷 (Lnatural) 识别电网的峰区与谷区，
并严格按照论文公式(11)与(12)计算各时段的充放电调度概率分布。
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List

def identify_peak_valley_hours(
    df_load: pd.DataFrame, 
    load_col: str = 'Lnatural', 
    time_col: str = 'Hour', 
    n_hours: int = 5
) -> Tuple[List[int], List[int], List[int]]:
    """
    识别电网负荷的峰区、谷区与平段。
    依据论文设定：取自然电力负荷排名前五的时间区间为峰区，后五为谷区。
    
    :param df_load: 包含 24 小时自然负荷的 DataFrame
    :param load_col: 负荷数值所在的列名
    :param time_col: 时间(小时)所在的列名
    :param n_hours: 峰区/谷区各自包含的小时数 (默认 5)
    :return: (峰区小时列表, 谷区小时列表, 平段小时列表)
    """
    if len(df_load) != 24:
        raise ValueError(f"输入的负荷数据应包含 24 小时的数据，当前长度为 {len(df_load)}")

    # 提取前 n_hours 个最大负荷作为峰区
    peak_df = df_load.nlargest(n_hours, load_col)
    peak_hours = peak_df[time_col].astype(int).tolist()
    
    # 提取前 n_hours 个最小负荷作为谷区
    valley_df = df_load.nsmallest(n_hours, load_col)
    valley_hours = valley_df[time_col].astype(int).tolist()
    
    # 剩余的 14 个小时即为平段 (C类用户出发时间如果在平段，可参与V2G)
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
    """
    计算峰区和谷区各时段的充电概率分布。
    
    - 峰区 (公式11): p_i = L_i / sum(L_j)，按负荷正向比例分配 (用于C类反向放电的时间权重)
    - 谷区 (公式12): p_i = (1/L_i) / sum(1/L_j)，按负荷逆向加权分配 (用于A类谷区充电的时间权重)
    
    :param df_load: 包含 24 小时自然负荷的 DataFrame
    :param peak_hours: 峰区小时列表
    :param valley_hours: 谷区小时列表
    :param load_col: 负荷数值所在的列名
    :param time_col: 时间所在的列名
    :return: (峰区各小时的概率字典, 谷区各小时的概率字典)
    """
    df_indexed = df_load.set_index(time_col)
    
    # 1. 计算峰区概率 (正向比例)
    peak_loads = df_indexed.loc[peak_hours, load_col]
    peak_total = peak_loads.sum()
    peak_probs = (peak_loads / peak_total).round(4).to_dict()
    
    # 2. 计算谷区概率 (逆向加权 - 论文公式12)
    # 负荷越小，其倒数越大，分配的概率(权重)就越高，填谷效果越好
    valley_loads = df_indexed.loc[valley_hours, load_col]
    valley_inv_loads = 1.0 / valley_loads
    valley_inv_total = valley_inv_loads.sum()
    valley_probs = (valley_inv_loads / valley_inv_total).round(4).to_dict()
    
    # 校验概率和是否为 1 (容忍浮点数舍入误差)
    assert np.isclose(sum(peak_probs.values()), 1.0, atol=1e-3), "峰区概率和不为1"
    assert np.isclose(sum(valley_probs.values()), 1.0, atol=1e-3), "谷区概率和不为1"
    
    return peak_probs, valley_probs

def build_grid_context(df_load: pd.DataFrame, load_col: str = 'Lnatural') -> dict:
    """
    核心执行函数：整合上述步骤，为 V2G 模拟提供电网背景上下文 (Context)。
    
    :param df_load: 包含 24 小时自然负荷的 DataFrame
    :param load_col: 负荷列名
    :return: 包含峰谷时段划分及相应概率的字典结构
    """
    # 确保时间列名为 Hour，以防不一致
    if 'hour' in df_load.columns and 'Hour' not in df_load.columns:
        df_load = df_load.rename(columns={'hour': 'Hour'})
        
    peak_hrs, valley_hrs, flat_hrs = identify_peak_valley_hours(df_load, load_col, time_col='Hour')
    peak_probs, valley_probs = calculate_charging_probabilities(df_load, peak_hrs, valley_hrs, load_col, time_col='Hour')
    
    # 为了方便蒙特卡洛 np.random.choice 抽样，将概率字典转换为两个对齐的列表
    # 峰区抽样参数
    peak_choices = list(peak_probs.keys())
    peak_weights = list(peak_probs.values())
    
    # 谷区抽样参数
    valley_choices = list(valley_probs.keys())
    valley_weights = list(valley_probs.values())
    
    return {
        "peak_hours": peak_hrs,
        "valley_hours": valley_hrs,
        "flat_hours": flat_hrs,
        "peak_sampling": {
            "hours": peak_choices,
            "probs": peak_weights
        },
        "valley_sampling": {
            "hours": valley_choices,
            "probs": valley_weights
        }
    }

# ==========================================
# 测试代码 (模块内独立运行时执行)
# ==========================================
if __name__ == "__main__":
    # 模拟生成一个 24 小时的城市自然负荷数据用于测试
    test_load_data = pd.DataFrame({
        'Hour': list(range(24)),
        # 简单模拟：白天低，晚上高，深夜极低
        'Lnatural': [2000, 1900, 1800, 1850, 2100, 2500, 3000, 4000, 4500, 4200, 
                     3800, 3500, 3600, 4000, 4800, 5500, 6000, 6500, 6800, 7000, 
                     6900, 6200, 5000, 3000]
    })
    
    grid_ctx = build_grid_context(test_load_data)
    
    print("【电网负荷分析结果】")
    print(f"峰区时间 (Top 5): {grid_ctx['peak_hours']}")
    print(f" -> 峰区充电概率分配: {dict(zip(grid_ctx['peak_sampling']['hours'], grid_ctx['peak_sampling']['probs']))}")
    
    print(f"\n谷区时间 (Bottom 5): {grid_ctx['valley_hours']}")
    print(f" -> 谷区充电概率分配 (逆向加权): {dict(zip(grid_ctx['valley_sampling']['hours'], grid_ctx['valley_sampling']['probs']))}")
    
    print(f"\n平段时间: {grid_ctx['flat_hours']}")