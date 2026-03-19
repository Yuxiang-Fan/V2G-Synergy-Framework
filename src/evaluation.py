"""
综合评估与协同效应分析模块 (evaluation.py)
负责计算有序充电策略下的各项评估指标（削峰提谷率、负荷方差、负荷率等），
并利用多元线性回归模型（OLS）执行跨城市的双因子协同效应分析。
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings

def calculate_evaluation_metrics(natural_load: np.ndarray, dispatch_load: np.ndarray) -> dict:
    """
    计算基于表4的电动汽车有序充电策略评估指标。
    
    :param natural_load: 长度为 24 的自然电力负荷数组 (L_natural)
    :param dispatch_load: 长度为 24 的调度后电力负荷数组 (L_dispatch)
    :return: 包含各项评估指标的字典
    """
    if len(natural_load) != 24 or len(dispatch_load) != 24:
        raise ValueError("输入的负荷数组长度必须为 24。")
        
    # 自然负荷统计量
    peak_natural = np.max(natural_load)
    valley_natural = np.min(natural_load)
    mean_natural = np.mean(natural_load)
    
    # 调度后负荷统计量
    peak_dispatch = np.max(dispatch_load)
    valley_dispatch = np.min(dispatch_load)
    mean_dispatch = np.mean(dispatch_load)
    
    # 1. 峰值削减率 (Reduction_dispatch)
    # 公式: (1 - Peak_dispatch / Peak_natural) * 100%
    peak_reduction_rate = (1.0 - peak_dispatch / peak_natural) * 100.0
    
    # 2. 谷值提升率 (Improvement_dispatch)
    # 公式: (Valley_dispatch / Valley_natural - 1) * 100%
    valley_improvement_rate = (valley_dispatch / valley_natural - 1.0) * 100.0
    
    # 3. 电力负荷峰谷差 (ΔL)
    delta_L_natural = peak_natural - valley_natural
    delta_L_dispatch = peak_dispatch - valley_dispatch
    
    # 4. 电力负荷率 (LR)
    # 公式: (L_mean / L_peak) * 100%
    lr_natural = (mean_natural / peak_natural) * 100.0
    lr_dispatch = (mean_dispatch / peak_dispatch) * 100.0
    
    # 5. 电力负荷方差 (σ²)
    var_natural = np.var(natural_load, ddof=0)
    var_dispatch = np.var(dispatch_load, ddof=0)
    
    return {
        "Peak_Natural": peak_natural,
        "Peak_Dispatch": peak_dispatch,
        "Valley_Natural": valley_natural,
        "Valley_Dispatch": valley_dispatch,
        "Peak_Reduction_Rate_%": peak_reduction_rate,
        "Valley_Improvement_Rate_%": valley_improvement_rate,
        "Delta_L_Natural": delta_L_natural,
        "Delta_L_Dispatch": delta_L_dispatch,
        "Load_Ratio_Natural_%": lr_natural,
        "Load_Ratio_Dispatch_%": lr_dispatch,
        "Variance_Natural": var_natural,
        "Variance_Dispatch": var_dispatch
    }

def analyze_dual_factor_synergy(
    city_names: list,
    peak_reduction_rates: list,
    delta_L_naturals: list,
    v2g_vehicle_counts: list,
    city_populations: list
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    执行双因子协同效应分析 (公式 25, 26, 27)。
    利用多元线性回归评估对数化人均日负荷峰谷差与对数化人均V2G潜力对调峰效能的影响。
    
    :param city_names: 城市名称列表 (如 ['New York', 'Los Angeles', ...])
    :param peak_reduction_rates: 各城市对应的峰值削减率 (作为因变量 m_i)
    :param delta_L_naturals: 各城市对应的自然日负荷峰谷差
    :param v2g_vehicle_counts: 各城市具备双向充放电功能的电动汽车数 (V_V2G)
    :param city_populations: 各城市常住人口数 (POP)
    :return: statsmodels 的回归结果对象，可调用 .summary() 打印详细报告
    """
    # 转换为 NumPy 数组以便于向量化计算
    m_i = np.array(peak_reduction_rates)
    delta_L = np.array(delta_L_naturals)
    v2g_counts = np.array(v2g_vehicle_counts)
    pop = np.array(city_populations)
    
    # 公式(25): 构建对数化人均日负荷峰谷差指标 Ln(ΔL/POP)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a_i = np.log(delta_L / pop)
        
        # 公式(26): 构建对数化人均V2G调度潜力指标 Ln(V2G/POP)
        b_i = np.log(v2g_counts / pop)
    
    # 检查是否存在无效值 (如对负数或0求对数产生的 NaN/Inf)
    if np.any(np.isnan(a_i)) or np.any(np.isinf(a_i)) or np.any(np.isnan(b_i)) or np.any(np.isinf(b_i)):
        raise ValueError("对数化计算中产生了无效值 (NaN 或 Inf)，请检查输入数据是否包含负值或零。")
        
    # 构建多元线性规划模型数据结构
    # 因变量 Y
    Y = m_i
    
    # 自变量 X (添加常数项截距 beta_0)
    X = pd.DataFrame({
        'Ln(Delta_L/POP)': a_i,
        'Ln(V2G/POP)': b_i
    })
    X = sm.add_constant(X)
    
    # 公式(27): m_i = beta_0 + beta_1 * a_i + beta_2 * b_i + epsilon_i
    # 利用 OLS (普通最小二乘法) 拟合模型
    model = sm.OLS(Y, X)
    results = model.fit()
    
    return results

# ==========================================
# 测试代码 (模块内独立运行时执行)
# ==========================================
if __name__ == "__main__":
    # 1. 测试评估指标计算
    print("=== 测试单城市调度指标计算 ===")
    test_natural = np.array([2000, 1900, 1800, 1850, 2100, 2500, 3000, 4000, 4500, 4200, 
                             3800, 3500, 3600, 4000, 4800, 5500, 6000, 6500, 6800, 7000, 
                             6900, 6200, 5000, 3000])
    # 模拟一个优秀的调度结果：削峰填谷
    test_dispatch = test_natural.copy()
    test_dispatch[19] -= 500  # 峰值 7000 -> 6500
    test_dispatch[2] += 400   # 谷值 1800 -> 2200
    
    metrics = calculate_evaluation_metrics(test_natural, test_dispatch)
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")
        
    # 2. 测试双因子协同回归模型
    print("\n=== 测试双因子协同效应分析 (OLS 回归) ===")
    # 虚构 5 座城市的数据用于测试代码跑通
    test_cities = ["New York", "Los Angeles", "Houston", "San Diego", "San Francisco"]
    test_reductions = [5.2, 8.1, 4.5, 9.3, 7.6] # 削减率 %
    test_delta_Ls = [5200, 6500, 4800, 3100, 2900]
    test_v2gs = [189440, 282829, 92519, 99183, 28307] # 参考表1数据
    test_pops = [8330000, 3840000, 2300000, 1380000, 800000] 
    
    try:
        regression_res = analyze_dual_factor_synergy(
            test_cities, test_reductions, test_delta_Ls, test_v2gs, test_pops
        )
        print(regression_res.summary())
    except Exception as e:
        print(f"回归分析出错: {e}")