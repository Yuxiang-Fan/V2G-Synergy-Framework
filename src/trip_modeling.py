"""
出行时空特征建模模块 (trip_modeling.py)
负责从 NHTS 出行链数据中提取统计特征，并拟合出发时间(T)与行驶距离(D)的边缘分布。
为后续的 Copula 联合分布建模提供基础输入。
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import warnings

def calculate_silverman_bandwidth(data: np.ndarray) -> float:
    """
    使用 Silverman 经验法则 (Silverman's rule of thumb) 计算核密度估计的自适应带宽。
    公式: h = 0.9 * min(std, IQR / 1.34) * n^(-1/5)
    
    :param data: 一维连续数据数组 (如行驶距离)
    :return: 计算得到的平滑带宽
    """
    n = len(data)
    if n == 0:
        return 1.0
        
    std_dev = np.std(data, ddof=1)
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    
    # 取标准差和四分位距缩放值中的较小者，防止异常值拉大带宽
    min_dispersion = min(std_dev, iqr / 1.34)
    
    # 极少数情况下分散度为0，采用标准差或给一个微小默认值
    if min_dispersion == 0:
        min_dispersion = std_dev if std_dev > 0 else 1.0
        
    bandwidth = 0.9 * min_dispersion * (n ** (-0.2))
    return float(bandwidth)


def fit_departure_time_distribution(departure_hours: np.ndarray) -> np.ndarray:
    """
    计算出发时间的 24 点离散经验分布 (概率质量函数 PMF)。
    依据论文设定：出发时间 T 服从 24 点离散经验分布，由 NHTS 数据直接估计。
    
    :param departure_hours: 出发时间数组，取值范围应为整数 0-23
    :return: 长度为 24 的 numpy 数组，表示各小时的出行概率
    """
    # 确保时间在 0-23 之间
    valid_hours = departure_hours[(departure_hours >= 0) & (departure_hours <= 23)].astype(int)
    
    if len(valid_hours) == 0:
        warnings.warn("有效的出发时间数据为空，返回均匀分布。")
        return np.ones(24) / 24.0
        
    # 计算每个小时的频次
    counts = np.bincount(valid_hours, minlength=24)
    
    # 归一化为概率
    pmf = counts / counts.sum()
    return pmf


def fit_trip_distance_kde(distances: np.ndarray) -> KernelDensity:
    """
    采用核密度估计 (KDE) 拟合连续的行驶距离分布。
    依据论文设定：使用 Epanechnikov 核函数，带宽通过 Silverman 准则自适应确定。
    
    :param distances: 行驶距离数组 (单位: miles)
    :return: 训练好的 sklearn KernelDensity 模型实例
    """
    # 过滤掉负数距离
    valid_distances = distances[distances >= 0]
    
    if len(valid_distances) == 0:
        raise ValueError("没有有效的行驶距离数据用于 KDE 拟合。")
        
    # 计算自适应带宽
    bw = calculate_silverman_bandwidth(valid_distances)
    
    # 实例化并拟合 KDE 模型 (Epanechnikov 核)
    kde_model = KernelDensity(kernel='epanechnikov', bandwidth=bw)
    
    # sklearn 的 KDE 需要 2D 数组输入格式 (n_samples, n_features)
    kde_model.fit(valid_distances.reshape(-1, 1))
    
    return kde_model


def extract_trip_statistics(df_trips: pd.DataFrame) -> dict:
    """
    提取并打印出行数据的基本统计特征 (平均值、中位数、四分位数等)。
    用于在日志或 Notebook 中验证数据分布是否符合"右偏分布"和"短途主导"特性。
    
    :param df_trips: 包含 'Departure_Hour' 和 'Trip_Distance_Miles' 的 DataFrame
    :return: 包含统计信息的字典
    """
    distances = df_trips['Trip_Distance_Miles'].dropna().values
    
    stats = {
        'count': len(distances),
        'mean': np.mean(distances),
        'std': np.std(distances, ddof=1),
        'median': np.median(distances),
        'q1': np.percentile(distances, 25),
        'q3': np.percentile(distances, 75),
        'max': np.max(distances),
        'pct_under_5_miles': np.mean(distances <= 5.0) * 100,
        'pct_5_to_30_miles': np.mean((distances > 5.0) & (distances <= 30.0)) * 100,
        'pct_over_30_miles': np.mean(distances > 30.0) * 100
    }
    
    return stats


def build_marginal_distributions(df_trips: pd.DataFrame) -> dict:
    """
    核心执行函数：基于清洗后的出行链数据，构建出行时间与行驶距离的边缘分布模型。
    
    :param df_trips: 包含 'Departure_Hour' 和 'Trip_Distance_Miles' 的 DataFrame
    :return: 包含时间 PMF、距离 KDE 模型及距离基础数据的字典
    """
    if 'Departure_Hour' not in df_trips.columns or 'Trip_Distance_Miles' not in df_trips.columns:
        raise ValueError("输入的 DataFrame 必须包含 'Departure_Hour' 和 'Trip_Distance_Miles' 列。")
        
    # 1. 拟合出发时间经验分布
    departure_hours = df_trips['Departure_Hour'].values
    time_pmf = fit_departure_time_distribution(departure_hours)
    
    # 2. 拟合行驶距离连续分布 (KDE)
    trip_distances = df_trips['Trip_Distance_Miles'].values
    distance_kde = fit_trip_distance_kde(trip_distances)
    
    # 3. 提取特征统计量用于日志记录
    stats = extract_trip_statistics(df_trips)
    
    return {
        'time_pmf': time_pmf,
        'distance_kde': distance_kde,
        'statistics': stats,
        'raw_distances': trip_distances[trip_distances >= 0] # 传递有效原数据给 Copula 拟合使用
    }

# 测试代码 (如果在模块内直接运行)
if __name__ == "__main__":
    # 构造少量模拟数据进行测试
    sim_data = pd.DataFrame({
        'Departure_Hour': np.random.randint(0, 24, 1000),
        'Trip_Distance_Miles': np.random.exponential(16.0, 1000) # 模拟右偏分布
    })
    
    results = build_marginal_distributions(sim_data)
    print("时间经验概率 (前5个小时):", results['time_pmf'][:5])
    print("距离统计特征:", results['statistics'])
    print("KDE 带宽:", results['distance_kde'].bandwidth)