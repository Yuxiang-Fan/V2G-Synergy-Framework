import numpy as np
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
import warnings

class FrankCopulaModel:
    """Frank Copula 数学模型核心实现"""
    def __init__(self, theta: float = -0.637):
        """初始化模型参数，默认使用实证研究推荐值"""
        self.theta = float(theta)

    def log_density(self, u: np.ndarray, v: np.ndarray, theta: float = None) -> np.ndarray:
        """计算 Frank Copula 对数密度函数 ln c_theta(u, v)"""
        th = theta if theta is not None else self.theta

        if abs(th) < 1e-5:
            return np.zeros_like(u)

        u = np.clip(u, 1e-6, 1.0 - 1e-6)
        v = np.clip(v, 1e-6, 1.0 - 1e-6)

        num = -th * (np.exp(-th) - 1) * np.exp(-th * (u + v))
        den_term1 = np.exp(-th) - 1
        den_term2 = (np.exp(-th * u) - 1) * (np.exp(-th * v) - 1)
        den = (den_term1 + den_term2) ** 2

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c_uv = num / den
            log_c = np.log(np.clip(c_uv, 1e-10, None))

        return log_c

    def fit_mle(self, u: np.ndarray, v: np.ndarray) -> float:
        """采用 MLE 拟合最优 theta 参数"""
        def neg_log_likelihood(th):
            if abs(th) < 1e-5:
                return 1e10 
            return -np.sum(self.log_density(u, v, th))

        res = minimize_scalar(neg_log_likelihood, bounds=(-20, 20), method='bounded')

        if res.success:
            self.theta = res.x
            return self.theta
        else:
            raise ValueError("Frank Copula MLE fitting failed.")

    def sample_v_given_u(self, u: np.ndarray, n_samples: int = None) -> np.ndarray:
        """条件抽样：利用 Frank Copula 解析逆生成 V ~ F_{V|U}(v|u)"""
        th = self.theta

        if np.isscalar(u):
            u_arr = np.full(n_samples if n_samples else 1, u)
        else:
            u_arr = np.asarray(u)
            if n_samples is not None and len(u_arr) != n_samples:
                raise ValueError("Array u length mismatch with n_samples")

        w = np.random.uniform(0, 1, size=len(u_arr))

        if abs(th) < 1e-5:
            return w 

        exp_th = np.exp(-th)
        exp_thu = np.exp(-th * u_arr)

        term1 = w * (exp_th - 1)
        term2 = exp_thu * (1 - w) + w

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ratio = term1 / term2
            v = -(1.0 / th) * np.log(np.clip(1.0 + ratio, 1e-10, None))

        return np.clip(v, 0.0, 1.0)


class SpatiotemporalDependenceModel:
    """基于 Frank Copula 的时空联合分布建模与抽样引擎"""
    def __init__(self, time_pmf: np.ndarray, distance_kde, theta: float = -0.637):
        """融合出发时间经验概率与行驶距离 KDE 模型"""
        self.time_pmf = np.asarray(time_pmf)
        self.time_cdf = np.cumsum(self.time_pmf)
        self.time_cdf /= self.time_cdf[-1] 

        self.kde = distance_kde
        self.copula = FrankCopulaModel(theta=theta)

        self._build_distance_interpolators()

    def _build_distance_interpolators(self):
        """数值积分构建距离 KDE 的 CDF 与逆映射插值器"""
        d_grid = np.linspace(0, 1000, 5000)

        log_pdf = self.kde.score_samples(d_grid.reshape(-1, 1))
        pdf = np.exp(log_pdf)

        cdf = np.cumsum(pdf) * (d_grid[1] - d_grid[0])
        cdf = cdf / cdf[-1] 
        cdf = np.maximum.accumulate(cdf)

        self.f_d_interp = interp1d(d_grid, cdf, kind='linear', bounds_error=False, fill_value=(0.0, 1.0))
        self.f_inv_d_interp = interp1d(cdf, d_grid, kind='linear', bounds_error=False, fill_value=(0.0, 1000.0))

    def train_copula_parameter(self, raw_times: np.ndarray, raw_distances: np.ndarray) -> float:
        """利用原始时空数据通过 MLE 重拟合 Copula 相关性参数"""
        u_data = self.time_cdf[raw_times.astype(int)]
        v_data = self.f_d_interp(raw_distances)

        return self.copula.fit_mle(u_data, v_data)

    def sample_distance_given_time(self, departure_times: np.ndarray) -> np.ndarray:
        """核心业务接口：基于出发时间约束，经 Copula 映射抽样对应的行驶距离"""
        dep_times = np.clip(np.asarray(departure_times).astype(int), 0, 23)
        
        u_arr = self.time_cdf[dep_times]
        v_arr = self.copula.sample_v_given_u(u_arr)
        
        return self.f_inv_d_interp(v_arr)
