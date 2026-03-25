import numpy as np
import pandas as pd
import statsmodels.api as sm

def calculate_evaluation_metrics(natural_load: np.ndarray, dispatch_load: np.ndarray) -> dict:
    """计算有序充电策略的核心评估指标"""
    if len(natural_load) != 24 or len(dispatch_load) != 24:
        raise ValueError("输入负荷数组长度必须为24小时")
        
    peak_natural = np.max(natural_load)
    valley_natural = np.min(natural_load)
    mean_natural = np.mean(natural_load)
    
    peak_dispatch = np.max(dispatch_load)
    valley_dispatch = np.min(dispatch_load)
    mean_dispatch = np.mean(dispatch_load)
    
    return {
        "Peak_Natural": peak_natural,
        "Peak_Dispatch": peak_dispatch,
        "Valley_Natural": valley_natural,
        "Valley_Dispatch": valley_dispatch,
        "Peak_Reduction_Rate_%": (1.0 - peak_dispatch / peak_natural) * 100.0,
        "Valley_Improvement_Rate_%": (valley_dispatch / valley_natural - 1.0) * 100.0,
        "Delta_L_Natural": peak_natural - valley_natural,
        "Delta_L_Dispatch": peak_dispatch - valley_dispatch,
        "Load_Ratio_Natural_%": (mean_natural / peak_natural) * 100.0,
        "Load_Ratio_Dispatch_%": (mean_dispatch / peak_dispatch) * 100.0,
        "Variance_Natural": np.var(natural_load, ddof=0),
        "Variance_Dispatch": np.var(dispatch_load, ddof=0)
    }

def analyze_dual_factor_synergy(
    peak_reduction_rates: list,
    delta_L_naturals: list,
    v2g_vehicle_counts: list,
    city_populations: list
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """执行双因子协同效应 OLS 回归分析"""
    Y = np.array(peak_reduction_rates)
    delta_L = np.array(delta_L_naturals)
    v2g_counts = np.array(v2g_vehicle_counts)
    pop = np.array(city_populations)
    
    # 构建对数化人均自变量
    X = pd.DataFrame({
        'Ln_Delta_L_per_capita': np.log(delta_L / pop),
        'Ln_V2G_per_capita': np.log(v2g_counts / pop)
    })
    
    if X.isin([np.nan, np.inf, -np.inf]).any().any():
        raise ValueError("对数化计算产生无效值，请检查输入数据是否存在非正数")
        
    # 添加常数项截距并拟合 OLS 模型
    X = sm.add_constant(X)
    model = sm.OLS(Y, X)
    
    return model.fit()

if __name__ == "__main__":
    test_natural = np.array([2000, 1900, 1800, 1850, 2100, 2500, 3000, 4000, 4500, 4200, 
                             3800, 3500, 3600, 4000, 4800, 5500, 6000, 6500, 6800, 7000, 
                             6900, 6200, 5000, 3000])
    
    test_dispatch = test_natural.copy()
    test_dispatch[19] -= 500 
    test_dispatch[2] += 400  
    
    metrics = calculate_evaluation_metrics(test_natural, test_dispatch)
    print("=== 调度评估指标 ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")
        
    test_reductions = [5.2, 8.1, 4.5, 9.3, 7.6] 
    test_delta_Ls = [5200, 6500, 4800, 3100, 2900]
    test_v2gs = [189440, 282829, 92519, 99183, 28307] 
    test_pops = [8330000, 3840000, 2300000, 1380000, 800000] 
    
    regression_res = analyze_dual_factor_synergy(
        test_reductions, test_delta_Ls, test_v2gs, test_pops
    )
    
    print("\n=== OLS 回归分析报告 ===")
    print(regression_res.summary())
