import os
import numpy as np

# --- 数据路径与城市配置 ---
# 环境变量获取外部数据根目录
EXTERNAL_DATA_ROOT = os.getenv("V2G_EXTERNAL_DATA_ROOT", "/path/to/your/external/datasets")

DATA_PATHS = {
    "eia_load_data": os.path.join(EXTERNAL_DATA_ROOT, "EIA_Load_2025.csv"),
    "ev_registration_data": os.path.join(EXTERNAL_DATA_ROOT, "EV_Registrations_2024.csv"),
    "nhts_trip_data": os.path.join(EXTERNAL_DATA_ROOT, "NHTS_Trips.csv"),
    "natural_load_base": os.path.join(EXTERNAL_DATA_ROOT, "natural_loads")
}

# 选定的五座美国代表性研究城市
TARGET_CITIES = ["New York", "Los Angeles", "Houston", "San Diego", "San Francisco"]

# --- 仿真全局控制参数 ---
NUM_SIMULATION_RUNS = 1000       # 每个实验组的蒙特卡洛迭代次数
NUM_PROCESSES = 64               # 并行计算核心数
RANDOM_SEED = 12345              # 全局随机种子

# 用户参与度序列
PARTICIPATION_RATES = np.linspace(0.1, 1.0, 10).tolist()

# --- EV 物理特性参数 ---
# 8类细分车型的电池容量配置（单位：kWh）
BATTERY_CAPACITY_MAP = {
    1: 150, 2: 150, 3: 85, 4: 85,
    5: 55,  6: 55,  7: 25, 8: 25
}

# 8类细分车型的单位能耗配置（单位：kWh/100km）
# 奇数代表磷酸铁锂电池，偶数代表三元锂电池
ENERGY_CONSUMPTION_MAP = {
    1: 17, 2: 20, 3: 17, 4: 20,
    5: 17, 6: 20, 7: 17, 8: 20
}

# --- 用户行为概率分布 ---
# 初始 SOC 状态的 Beta 分布参数
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

# 单日充电次数概率分布
CHARGING_TIMES_BINS = [0, 1, 2]
CHARGING_TIMES_PROBS = [0.6350, 0.2960, 0.0690]

# 行驶里程离散抽样区间与概率
MILEAGE_BINS = [0, 15, 137, 242, 349, 431, 548, 671, 1592, 1778]
MILEAGE_PROBS = [0.4013, 0.5766, 0.0109, 0.0033, 0.0025, 0.0007, 0.0007, 0.0018, 0.0012, 0.0010]

# 24小时出发时间基准概率分布
BASE_DEPARTURE_PROBS = [
    0.08, 0.08, 0.01, 0.01, 0.01, 0.01, 0.01, 0.025, 0.03, 0.04, 0.05, 0.05,
    0.05, 0.05, 0.05, 0.05, 0.04, 0.03, 0.025, 0.02, 0.01, 0.08, 0.08, 0.08
]

# --- V2G 调度与约束参数 ---
# 默认峰谷时段配置
DEFAULT_PEAK_HOURS = [0, 1, 21, 22, 23]
DEFAULT_TROUGH_HOURS = [7, 8, 9, 10, 11]

# 谷区自然充电行为权重
TROUGH_CHARGING_BASE_PROBS = [0.2032, 0.1987, 0.1968, 0.1978, 0.2035]

# 峰区有序充电权重
PEAK_CHARGING_BASE_PROBS = [0.2026, 0.1999, 0.1949, 0.1989, 0.2037]

# V2G 充放电 SOC 约束
V2G_SOC_CONSTRAINTS = {
    "forward_charge_min_soc": 0.9,     # 填谷充电后的目标电量下限
    "reverse_charge_range": (0.2, 0.3) # 削峰放电后的电量保留区间
}

# 辅助函数：标准化概率数组
def normalize_probs(probs):
    arr = np.array(probs, dtype=float)
    return (arr / arr.sum()).tolist()

# 静态校验并初始化概率参数
CHARGING_TIMES_PROBS = normalize_probs(CHARGING_TIMES_PROBS)
MILEAGE_PROBS = normalize_probs(MILEAGE_PROBS)
BASE_DEPARTURE_PROBS = normalize_probs(BASE_DEPARTURE_PROBS)
TROUGH_CHARGING_BASE_PROBS = normalize_probs(TROUGH_CHARGING_BASE_PROBS)
PEAK_CHARGING_BASE_PROBS = normalize_probs(PEAK_CHARGING_BASE_PROBS)
