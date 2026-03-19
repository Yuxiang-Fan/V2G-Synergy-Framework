"""
数据加载与预处理模块 (data_loader.py)
负责读取 EIA电力负荷、EV注册数据、NHTS出行特征等外部数据，
并执行异常值检测、KNN插值、中位数聚法等预处理工作。
"""

import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import warnings

# 引入全局配置
from src.config import DATA_PATHS, TARGET_CITIES

warnings.filterwarnings('ignore', category=UserWarning)

def clean_and_impute_load_data(df: pd.DataFrame, load_col: str = 'Load') -> pd.DataFrame:
    """
    对电力负荷数据进行异常值检测与插补。
    依据论文方法：改进型箱线图法检测异常值，KNN(k=5)算法进行邻域插补。
    
    :param df: 包含时序负荷数据的 DataFrame
    :param load_col: 负荷数据所在的列名
    :return: 清洗插补后的 DataFrame
    """
    data = df.copy()
    
    # 1. 缺失值与异常值检测 (改进型箱线图法)
    Q1 = data[load_col].quantile(0.25)
    Q3 = data[load_col].quantile(0.75)
    IQR = Q3 - Q1
    
    # 设定异常阈值 (通常为 1.5 倍 IQR)
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 将超出阈值的异常点标记为 NaN
    outlier_mask = (data[load_col] < lower_bound) | (data[load_col] > upper_bound)
    data.loc[outlier_mask, load_col] = np.nan
    
    # 2. KNN 邻域插补 (k=5)
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    
    # KNNImputer 需要 2D array，这里我们将负荷列重塑后进行插值
    load_values_reshaped = data[[load_col]].values
    imputed_values = imputer.fit_transform(load_values_reshaped)
    
    data[load_col] = imputed_values.flatten()
    
    return data

def get_city_synthesis_load(city_name: str, file_path: str = None) -> pd.DataFrame:
    """
    获取指定城市的综合电力负荷曲线 (Lsynthesis)。
    依据论文方法：读取长期(如全年)连续监测记录，清洗后采用中位数聚法构建代表性日负荷模式。
    
    :param city_name: 城市名称 (例如 "New York")
    :param file_path: EIA 数据文件路径 (默认从 config 读取)
    :return: 包含 0-23 小时及对应中位数负荷的 DataFrame
    """
    if file_path is None:
        file_path = DATA_PATHS.get("eia_load_data")
        
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到 EIA 负荷数据文件: {file_path}")
        
    # 假设外部 CSV 格式包含：['City', 'Date', 'Hour', 'Load']
    df_raw = pd.read_csv(file_path)
    
    # 筛选目标城市数据
    df_city = df_raw[df_raw['City'] == city_name].copy()
    if df_city.empty:
        raise ValueError(f"数据集中未找到城市 {city_name} 的记录。")
        
    # 清洗与插值
    df_cleaned = clean_and_impute_load_data(df_city, load_col='Load')
    
    # 中位数聚法：按小时分组取中位数，过滤极端天气/节假日短期波动
    synthesis_load = df_cleaned.groupby('Hour')['Load'].median().reset_index()
    synthesis_load.rename(columns={'Load': 'Lsynthesis'}, inplace=True)
    
    return synthesis_load

def get_city_ev_count(city_name: str, zip_codes: list, file_path: str = None) -> int:
    """
    获取特定城市的电动汽车保有量。
    依据论文方法：通过各州行政区划代码(ZIP Code)进行数据筛选和匹配获取统计量。
    
    :param city_name: 城市名称
    :param zip_codes: 该城市包含的 ZIP 码列表
    :param file_path: Atlas EV Hub/CEC 注册数据路径
    :return: 该城市的电动汽车总保有量
    """
    if file_path is None:
        file_path = DATA_PATHS.get("ev_registration_data")
        
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到电动汽车保有量数据文件: {file_path}")
        
    # 假设外部 CSV 格式包含：['ZIP_Code', 'Vehicle_Type', 'Make', 'Model']
    # 实际应用中根据源数据分块读取或整体读取
    df_ev = pd.read_csv(file_path, usecols=['ZIP_Code'])
    
    # 匹配城市 ZIP 码范围
    city_ev_data = df_ev[df_ev['ZIP_Code'].isin(zip_codes)]
    total_ev_count = len(city_ev_data)
    
    return total_ev_count

def get_nhts_trip_data(file_path: str = None) -> pd.DataFrame:
    """
    获取并清洗 NHTS 出行链数据，为 Copula 时空特征联合分布建模提供基础输入。
    
    :param file_path: NHTS 调查数据路径
    :return: 清洗后包含 ['Departure_Hour', 'Trip_Distance_Miles'] 的 DataFrame
    """
    if file_path is None:
        file_path = DATA_PATHS.get("nhts_trip_data")
        
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到 NHTS 出行特征数据文件: {file_path}")
        
    # 假设外部 CSV 格式包含出发时间和行驶距离
    df_trips = pd.read_csv(file_path, usecols=['STRTTIME', 'TRPMILES'])
    
    # 重命名便于后续处理
    df_trips.rename(columns={'STRTTIME': 'Departure_Time', 'TRPMILES': 'Trip_Distance_Miles'}, inplace=True)
    
    # 剔除无效值 (例如 NHTS 中用负数表示未作答)
    df_trips = df_trips[(df_trips['Departure_Time'] >= 0) & (df_trips['Trip_Distance_Miles'] >= 0)]
    
    # 将 HHMM 格式的时间转换为整点小时 (0-23)
    df_trips['Departure_Hour'] = (df_trips['Departure_Time'] // 100).astype(int)
    df_trips.loc[df_trips['Departure_Hour'] == 24, 'Departure_Hour'] = 0  # 将 24 点归至 0 点
    
    return df_trips[['Departure_Hour', 'Trip_Distance_Miles']].reset_index(drop=True)

def load_city_natural_load(city_name: str, base_dir: str = None) -> pd.DataFrame:
    """
    读取已经剥离了电动汽车基础充电耗电量的自然电力负荷 (Lnatural)。
    该数据用于作为 V2G 有序充电策略评估的“纯净基线”。
    
    :param city_name: 城市名称 (需与文件名匹配，例如 "New_York")
    :param base_dir: 存放 Lnatural 表格的外部文件夹路径
    :return: 包含 24 小时自然电力负荷的 DataFrame
    """
    if base_dir is None:
        base_dir = DATA_PATHS.get("natural_load_base")
        
    # 城市名做简单格式化以适配文件名，如 "New York" -> "New_York"
    formatted_city_name = city_name.replace(" ", "_")
    file_path = os.path.join(base_dir, f"{formatted_city_name}.xlsx")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到城市 {city_name} 的自然负荷基线文件: {file_path}")
        
    # 假设你的输出表包含 'Hour' 和 'Lnatural' (或你之前代码里的 '除电车用电')
    df_natural = pd.read_excel(file_path)
    
    # 统一列名以供后续分析使用
    if '除电车用电' in df_natural.columns:
        df_natural.rename(columns={'除电车用电': 'Lnatural'}, inplace=True)
    if 'hour' in df_natural.columns:
        df_natural.rename(columns={'hour': 'Hour'}, inplace=True)
        
    return df_natural