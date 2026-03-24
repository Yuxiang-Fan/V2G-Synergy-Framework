import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import warnings

def calculate_silverman_bandwidth(data: np.ndarray) -> float:
    """使用 Silverman 法则计算 KDE 自适应带宽"""
    n = len(data)
    if n == 0:
        return 1.0
        
    std_dev = np.std(data, ddof=1)
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    
    # 取标准差与 IQR 缩放值的较小者，增强对异常值的鲁棒性
    min_dispersion = min(std_dev, iqr / 1.34)
    if min_dispersion == 0:
        min_dispersion = std_dev if std_dev > 0 else 1.0
        
    return float(0.9 * min_dispersion * (n ** (-0.2)))


def fit_departure_time_distribution(departure_hours: np.ndarray) -> np.ndarray:
    """计算出发时间的 24 点离散 PMF"""
    valid_hours = departure_hours[(departure_hours >= 0) & (departure_hours <= 23)].astype(int)
    
    if len(valid_hours) == 0:
        warnings.warn("Valid departure hours empty, returning uniform distribution.")
        return np.ones(24) / 24.0
        
    counts = np.bincount(valid_hours, minlength=24)
    return counts / counts.sum()


def fit_trip_distance_kde(distances: np.ndarray) -> KernelDensity:
    """采用 Epanechnikov 核函数拟合行驶距离连续分布"""
    valid_distances = distances[distances >= 0]
    if len(valid_distances) == 0:
        raise ValueError("No valid distance data for KDE fitting.")
        
    bw = calculate_silverman_bandwidth(valid_distances)
    
    kde_model = KernelDensity(kernel='epanechnikov', bandwidth=bw)
    kde_model.fit(valid_distances.reshape(-1, 1))
    
    return kde_model


def extract_trip_statistics(df_trips: pd.DataFrame) -> dict:
    """提取出行距离关键统计量"""
    distances = df_trips['Trip_Distance_Miles'].dropna().values
    
    return {
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


def build_marginal_distributions(df_trips: pd.DataFrame) -> dict:
    """构建出行时间与距离的边缘分布模型入口"""
    if 'Departure_Hour' not in df_trips.columns or 'Trip_Distance_Miles' not in df_trips.columns:
        raise ValueError("Missing required columns in DataFrame.")
        
    trip_distances = df_trips['Trip_Distance_Miles'].values
    
    return {
        'time_pmf': fit_departure_time_distribution(df_trips['Departure_Hour'].values),
        'distance_kde': fit_trip_distance_kde(trip_distances),
        'statistics': extract_trip_statistics(df_trips),
        'raw_distances': trip_distances[trip_distances >= 0]
    }

if __name__ == "__main__":
    sim_data = pd.DataFrame({
        'Departure_Hour': np.random.randint(0, 24, 1000),
        'Trip_Distance_Miles': np.random.exponential(16.0, 1000)
    })
    
    res = build_marginal_distributions(sim_data)
    print("Time PMF (first 5):", res['time_pmf'][:5])
    print("Distance Stats:", res['statistics'])
    print("KDE Bandwidth:", res['distance_kde'].bandwidth)
