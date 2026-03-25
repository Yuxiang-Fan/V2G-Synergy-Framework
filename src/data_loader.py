import os
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import warnings

from src.config import DATA_PATHS, TARGET_CITIES

warnings.filterwarnings('ignore', category=UserWarning)

def clean_and_impute_load_data(df: pd.DataFrame, load_col: str = 'Load') -> pd.DataFrame:
    """基于 IQR 检测异常值并执行 KNN 邻域插补"""
    data = df.copy()
    
    Q1 = data[load_col].quantile(0.25)
    Q3 = data[load_col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_mask = (data[load_col] < lower_bound) | (data[load_col] > upper_bound)
    data.loc[outlier_mask, load_col] = np.nan
    
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    
    load_values_reshaped = data[[load_col]].values
    data[load_col] = imputer.fit_transform(load_values_reshaped).flatten()
    
    return data

def get_city_synthesis_load(city_name: str, file_path: str = None) -> pd.DataFrame:
    """提取目标城市负荷数据，经清洗后按小时进行中位数聚合生成 Lsynthesis"""
    if file_path is None:
        file_path = DATA_PATHS.get("eia_load_data")
        
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"EIA load data not found: {file_path}")
        
    df_raw = pd.read_csv(file_path)
    df_city = df_raw[df_raw['City'] == city_name].copy()
    
    if df_city.empty:
        raise ValueError(f"No records found for city: {city_name}")
        
    df_cleaned = clean_and_impute_load_data(df_city, load_col='Load')
    
    synthesis_load = df_cleaned.groupby('Hour')['Load'].median().reset_index()
    synthesis_load.rename(columns={'Load': 'Lsynthesis'}, inplace=True)
    
    return synthesis_load

def get_city_ev_count(city_name: str, zip_codes: list, file_path: str = None) -> int:
    """根据 ZIP Code 匹配提取特定城市的 EV 保有量"""
    if file_path is None:
        file_path = DATA_PATHS.get("ev_registration_data")
        
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"EV registration data not found: {file_path}")
        
    df_ev = pd.read_csv(file_path, usecols=['ZIP_Code'])
    city_ev_data = df_ev[df_ev['ZIP_Code'].isin(zip_codes)]
    
    return len(city_ev_data)

def get_nhts_trip_data(file_path: str = None) -> pd.DataFrame:
    """读取并清洗 NHTS 数据，提取 Departure_Hour 与 Trip_Distance_Miles"""
    if file_path is None:
        file_path = DATA_PATHS.get("nhts_trip_data")
        
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"NHTS trip data not found: {file_path}")
        
    df_trips = pd.read_csv(file_path, usecols=['STRTTIME', 'TRPMILES'])
    df_trips.rename(columns={'STRTTIME': 'Departure_Time', 'TRPMILES': 'Trip_Distance_Miles'}, inplace=True)
    
    df_trips = df_trips[(df_trips['Departure_Time'] >= 0) & (df_trips['Trip_Distance_Miles'] >= 0)]
    
    df_trips['Departure_Hour'] = (df_trips['Departure_Time'] // 100).astype(int)
    df_trips.loc[df_trips['Departure_Hour'] == 24, 'Departure_Hour'] = 0 
    
    return df_trips[['Departure_Hour', 'Trip_Distance_Miles']].reset_index(drop=True)

def load_city_natural_load(city_name: str, base_dir: str = None) -> pd.DataFrame:
    """读取剔除了 EV 基础耗电量的 Lnatural 纯净基线"""
    if base_dir is None:
        base_dir = DATA_PATHS.get("natural_load_base")
        
    formatted_city_name = city_name.replace(" ", "_")
    file_path = os.path.join(base_dir, f"{formatted_city_name}.xlsx")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Lnatural base file not found for {city_name}: {file_path}")
        
    df_natural = pd.read_excel(file_path)
    
    if '除电车用电' in df_natural.columns:
        df_natural.rename(columns={'除电车用电': 'Lnatural'}, inplace=True)
    if 'hour' in df_natural.columns:
        df_natural.rename(columns={'hour': 'Hour'}, inplace=True)
        
    return df_natural
