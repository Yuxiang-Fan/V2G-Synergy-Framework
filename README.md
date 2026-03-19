# Cross-Regional V2G Benefit Evaluation Framework

This repository provides a systematic evaluation framework for analyzing the benefits of Vehicle-to-Grid (V2G) technology. By integrating Monte Carlo simulations with spatiotemporal behavior modeling, the project evaluates the impact of electric vehicles (EVs)—acting as mobile distributed energy storage (MDES)—on urban power grid peak shaving and valley filling. The analysis is applied across five representative U.S. cities: New York, Los Angeles, Houston, San Diego, and San Francisco.

## Project Structure

* `src/`: Core module library
  * `config.py`: Global parameter configuration and external data path mapping.
  * `data_loader.py`: Data ingestion and preprocessing (e.g., IQR outlier detection, KNN imputation).
  * `trip_modeling.py`: Extraction of travel behavior statistics and marginal distribution fitting.
  * `copula_model.py`: Spatiotemporal dependence modeling utilizing the Frank Copula function.
  * `load_analysis.py`: Grid peak/valley identification and charging probability calculation.
  * `ev_simulation.py`: Baseline EV state initialization and uncoordinated charging simulation.
  * `v2g_strategy.py`: Classified V2G scheduling strategy (A/B/C groups) and parallel simulation engine.
  * `evaluation.py`: Calculation of metrics and execution of dual-factor synergy regression analysis.
* `scripts/`: Automated execution workflow (Sequential execution from 01 to 05).
* `results/`: Directory for output data, including baseline loads and regression reports.

## Data Sources

The framework utilizes recognized open-source datasets. Users must acquire these independently:

* **Grid Load Data**: Sourced from the **U.S. Energy Information Administration (EIA)**.
* **Travel Behavior Data**: Sourced from the **National Household Travel Survey (NHTS)**.
* **EV Population & Specifications**: Sourced from **Atlas EV Hub** and the **California Energy Commission (CEC)**.
* **Demographic Data**: Sourced from the **U.S. Census Bureau**.

## Methodology

The framework employs a three-tier approach:
1. **Spatiotemporal Dependency**: Applies Frank Copula theory to model travel distance and departure time joint distributions.
2. **Differentiated Scheduling**: Categorizes users (Peak, Valley, Flat) to execute targeted V2G dispatching.
3. **Synergy Evaluation**: Introduces a dual-factor model incorporating grid elasticity and EV penetration.

---

# 跨区域 V2G 效益评估框架

本项目提供了一个系统性的车辆到电网（V2G）效益评估框架。通过结合蒙特卡洛模拟与时空行为建模，项目旨在量化评估电动汽车作为移动分布式储能单元（MDES）对城市电网削峰填谷的贡献。研究框架应用于美国五座代表性城市：纽约、洛杉矶、休斯顿、圣地亚哥和旧金山。

## 目录结构

* `src/`: 核心模块库
  * `config.py`: 全局参数配置与数据路径映射。
  * `data_loader.py`: 数据加载与预处理（异常值检测、插补）。
  * `trip_modeling.py`: 出行行为统计特征提取与分布拟合。
  * `copula_model.py`: 基于 Frank Copula 函数的时空依赖建模。
  * `load_analysis.py`: 峰谷时段识别与充放电概率计算。
  * `ev_simulation.py`: 物理状态初始化与无序充电基线模拟。
  * `v2g_strategy.py`: 分类 V2G 调度策略与并行仿真引擎。
  * `evaluation.py`: 综合评估指标计算与双因子协同回归分析。
* `scripts/`: 自动化执行流（按 01-05 顺序运行）。
* `results/`: 输出目录，存放负荷基线、调度曲线与回归报告。

## 数据来源

本框架依托以下权威开源数据集，使用者需自行配置路径：

* **电网负荷数据**: 来源于美国能源信息署（EIA）。
* **出行行为数据**: 来源于全美家用出行调查（NHTS）。
* **电动汽车保有量及规格**: 来源于 Atlas EV Hub 与加州能源委员会（CEC）。
* **人口统计数据**: 来源于美国普查局（U.S. Census Bureau）。

## 研究方法

1. **时空依赖建模**: 引入 Frank Copula 理论构建出行距离与时间的联合分布。
2. **差异化调度策略**: 基于用户出行时间进行分类，执行针对性的 V2G 响应（如反向削峰）。
3. **协同效应评估**: 构建双因子协同模型，解析 V2G 效能在不同城市间的异质性成因。
