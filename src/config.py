"""
全局参数配置文件 (config.py)
用于存储跨区域V2G效益评估模型的所有静态参数、超参数以及外部数据路径配置。
"""

import os
import numpy as np

# =====================================================================
# 1. 外部数据源路径配置 (占位符，由使用者根据本地实际挂载或下载的路径配置)
# =====================================================================
# 建议使用者将开源数据（如EIA, NHTS, Atlas EV Hub等）统一放在一个外部目录
EXTERNAL_DATA_ROOT = os.getenv("V2G_EXTERNAL_DATA_ROOT", "/path/to/your/external/datasets")

DATA_PATHS = {
    # 美国能源信息署(EIA)电力负荷数据
    "eia_load_data": os.path.join(EXTERNAL_DATA_ROOT, "EIA_Load_2025.csv"),
    
    # Atlas EV Hub / 加州能源委员会 电动汽车保有量数据
    "ev_registration_data": os.path.join(EXTERNAL_DATA_ROOT, "EV_Registrations_2024.csv"),
    
    # 美国交通统计局(BTS) NHTS 出行链数据
    "nhts_trip_data": os.path.join(EXTERNAL_DATA_ROOT, "NHTS_Trips.csv"),
    
    # 各城市基础去除电车耗电后的自然负荷曲线 (由预处理脚本生成后指定的外部路径)
    "natural_load_base": os.path.join(EXTERNAL_DATA_ROOT, "natural_loads")
}

# 目标研究城市列表 (5座典型城市)
TARGET_CITIES = ["New York", "Los Angeles", "Houston", "San Diego", "San Francisco"]

# =====================================================================
# 2. 模拟运行超参数
# =====================================================================
# 蒙特卡洛模拟相关的全局控制参数
NUM_SIMULATION_RUNS = 1000       # 每个参与度下的模拟总次数
NUM_PROCESSES = 64               # 多进程并行数
RANDOM_SEED = 12345              # 全局随机种子，确保实验可复现

# 用户参与度序列 (从 10% 到 100%)
PARTICIPATION_RATES = np.linspace(0.1, 1.0, 10).tolist()

# =====================================================================
# 3. EV 物理特性参数 (电池与能耗)
# =====================================================================
# 电池分级与容量映射 (单位: kWh)
# 对应 8 种细分类型 (结合电池容量 150/85/55/25 与 正极材料差异)
BATTERY_CAPACITY_MAP = {
    1: 150, 
    2: 150, 
    3: 85,  
    4: 85,
    5: 55,  
    6: 55,  
    7: 25,  
    8: 25
}

# 电池单位能耗映射 (单位: kWh/100km)
# 三元锂电池(20) 与 磷酸铁锂电池(17)
ENERGY_CONSUMPTION_MAP = {
    1: 17, 
    2: 20, 
    3: 17, 
    4: 20,
    5: 17, 
    6: 20, 
    7: 17, 
    8: 20
}

# =====================================================================
# 4. 用户行为动态特征分布参数
# =====================================================================
# 初始荷电状态(SOC)的 Beta 分布参数 (alpha, beta)，基于中国特征数据库跨区域迁移得到
BETA_PARAMS_MAP = {
    1: (0.73, 0.41),
    2: (0.71, 0.40),
    3: (0.68, 0.45),
    4: (0.73, 0.41),
    5: (0.71, 0.40),
    6: (0.68, 0.45),
    7: (0.68, 0.45),
    8: (0.68, 0.45)
}

# 充电次数概率分布
CHARGING_TIMES_BINS = [0, 1, 2]
CHARGING_TIMES_PROBS = [0.6350, 0.2960, 0.0690]

# 离散化行驶里程区间与概率分布 (蒙特卡洛抽样用，Copula 模型使用连续分布)
MILEAGE_BINS = [0, 15, 137, 242, 349, 431, 548, 671, 1592, 1778]
MILEAGE_PROBS = [0.4013, 0.5766, 0.0109, 0.0033, 0.0025, 0.0007, 0.0007, 0.0018, 0.0012, 0.0010]

# 小时级充电时间(出发时间)基准概率分布 (长度为24，代表0点到23点)
BASE_DEPARTURE_PROBS = [
    0.08, 0.08, 0.01, 0.01, 0.01, 0.01, 0.01, 0.025, 0.03, 0.04, 0.05, 0.05,
    0.05, 0.05, 0.05, 0.05, 0.04, 0.03, 0.025, 0.02, 0.01, 0.08, 0.08, 0.08
]

# =====================================================================
# 5. 电网策略与 V2G 约束参数
# =====================================================================
# 静态设定的默认峰谷时段 (具体城市将由 load_analysis.py 动态重写覆盖)
DEFAULT_PEAK_HOURS = [0, 1, 21, 22, 23]
DEFAULT_TROUGH_HOURS = [7, 8, 9, 10, 11]

# 谷区自然充电行为的时间分布经验权重
TROUGH_CHARGING_BASE_PROBS = [0.2032, 0.1987, 0.1968, 0.1978, 0.2035]

# 峰区有序充电(正向)的时间分布经验权重
PEAK_CHARGING_BASE_PROBS = [0.2026, 0.1999, 0.1949, 0.1989, 0.2037]

# V2G 双向充电 SOC 约束设定
V2G_SOC_CONSTRAINTS = {
    "forward_charge_min_soc": 0.9,     # C类用户正向充电后的最低 SOC 要求
    "reverse_charge_range": (0.2, 0.3) # C类用户反向放电后的 SOC 保留区间
}

# 辅助函数：标准化概率数组，防止精度问题导致的报错
def normalize_probs(probs):
    arr = np.array(probs, dtype=float)
    return (arr / arr.sum()).tolist()

# 执行概率数组初始化校验
CHARGING_TIMES_PROBS = normalize_probs(CHARGING_TIMES_PROBS)
MILEAGE_PROBS = normalize_probs(MILEAGE_PROBS)
BASE_DEPARTURE_PROBS = normalize_probs(BASE_DEPARTURE_PROBS)
TROUGH_CHARGING_BASE_PROBS = normalize_probs(TROUGH_CHARGING_BASE_PROBS)
PEAK_CHARGING_BASE_PROBS = normalize_probs(PEAK_CHARGING_BASE_PROBS)