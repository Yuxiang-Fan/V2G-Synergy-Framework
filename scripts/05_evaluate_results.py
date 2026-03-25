import os
import sys
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import TARGET_CITIES, PARTICIPATION_RATES
from src.evaluation import analyze_dual_factor_synergy

CITY_POPULATION = {
    "New York": 8330000,
    "Los Angeles": 3840000,
    "Houston": 2300000,
    "San Diego": 1380000,
    "San Francisco": 800000
}

def export_marginal_differences(city: str, load_file_path: str, output_dir: str):
    """计算并导出不同参与度下负荷曲线的一阶与二阶差分 (边际效应)"""
    try:
        df_loads = pd.read_excel(load_file_path)
    except Exception as e:
        print(f"[Warning] 无法读取负荷文件 {city}: {e}")
        return

    pr_cols = [f"Ldispatch_{pr * 100:.0f}%" for pr in PARTICIPATION_RATES]
    valid_cols = [col for col in pr_cols if col in df_loads.columns]
    
    if len(valid_cols) < 3:
        print(f"[Skip] {city} 参与度样本不足，跳过边际差分计算。")
        return

    df_first_diff = pd.DataFrame({'Hour': df_loads['Hour']})
    for i in range(1, len(valid_cols)):
        df_first_diff[f"Δ1_{valid_cols[i]}"] = df_loads[valid_cols[i]] - df_loads[valid_cols[i - 1]]

    df_second_diff = pd.DataFrame({'Hour': df_loads['Hour']})
    for i in range(2, len(valid_cols)):
        # 二阶中心差分近似
        df_second_diff[f"Δ2_{valid_cols[i]}"] = (
            df_loads[valid_cols[i]] - 2.0 * df_loads[valid_cols[i - 1]] + df_loads[valid_cols[i - 2]]
        )

    output_path = os.path.join(output_dir, f"{city.replace(' ', '_')}_Marginal_Differences.xlsx")

    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df_loads.to_excel(writer, sheet_name='综合负荷曲线', index=False)
            df_first_diff.to_excel(writer, sheet_name='一阶边际增量', index=False)
            df_second_diff.to_excel(writer, sheet_name='二阶边际衰减', index=False)
        print(f"[+] {city} 边际效用差分计算完毕: {output_path}")
    except Exception as e:
        print(f"[Error] 保存差分结果失败 {city}: {e}")

def main():
    print("=== [05_evaluate_results] 启动综合评估与协同回归分析 ===")

    tables_dir = os.path.join(project_root, "results", "tables")
    if not os.path.exists(tables_dir):
        print(f"[Fatal Error] 结果目录缺失: {tables_dir}")
        sys.exit(1)

    print("\n[*] 计算各城市边际差分效应...")
    for city in TARGET_CITIES:
        load_file = os.path.join(tables_dir, f"{city.replace(' ', '_')}_V2G_Load_Curves.xlsx")
        if os.path.exists(load_file):
            export_marginal_differences(city, load_file, tables_dir)
        else:
            print(f"[Skip] 缺失负荷曲线: {city}")

    print("\n[*] 执行跨区域双因子协同效应 OLS 回归...")
    summary_file = os.path.join(tables_dir, "All_Cities_V2G_Metrics_Summary.csv")

    try:
        df_metrics = pd.read_csv(summary_file)
    except Exception as e:
        print(f"[Fatal Error] 读取指标汇总失败: {e}")
        sys.exit(1)

    max_pr = df_metrics['Participation_Rate'].max()
    df_max_pr = df_metrics[df_metrics['Participation_Rate'] == max_pr].copy()

    if len(df_max_pr) < 3:
        print("[Fatal Error] 城市样本不足，无法执行可靠的 OLS 回归。")
        sys.exit(1)

    m_i_list, delta_l_list, v2g_list, pop_list = [], [], [], []

    for _, row in df_max_pr.iterrows():
        c_name = row['City']
        if c_name in CITY_POPULATION:
            m_i_list.append(row['Peak_Reduction_Rate_%'])
            delta_l_list.append(row['Delta_L_Natural'])
            v2g_list.append(row['EV_Count'])
            pop_list.append(CITY_POPULATION[c_name])

    try:
        regression_results = analyze_dual_factor_synergy(
            peak_reduction_rates=m_i_list,
            delta_L_naturals=delta_l_list,
            v2g_vehicle_counts=v2g_list,
            city_populations=pop_list
        )

        print("\n" + "=" * 50)
        print("       双因子协同效应 OLS 回归分析报告")
        print("=" * 50)
        print(regression_results.summary())
        print("=" * 50)

        report_path = os.path.join(tables_dir, "Dual_Factor_Synergy_Regression_Report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("双因子协同效应 OLS 回归分析报告\n")
            f.write("模型设定: m_i = beta_0 + beta_1 * Ln(Delta_L / POP) + beta_2 * Ln(V2G / POP) + epsilon_i\n\n")
            f.write(regression_results.summary().as_text())
        print(f"\n[+] 回归报告已导出至: {report_path}")

    except Exception as e:
        print(f"\n[Error] 回归分析执行失败: {e}")

    print("\n=== [05_evaluate_results] 评估流程执行完毕 ===")

if __name__ == "__main__":
    main()
