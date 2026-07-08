
"""
lundon Hawkes Parameter Estimation

在已经拟合好的连续背景强度 μ(t) 的前提下，
对lundon火灾数据 (filtered_fire_incidents.csv) 进行 Hawkes 自激参数 (α, β) 的最大似然估计。
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from ld_possion import get_mu_t_parameters

# -------------------------------------------------------------------
# 配置路径
# -------------------------------------------------------------------

# 1) 你前面构造好的 μ(t) 模型（由 Shanghai_mu_t.py 生成）
#MU_MODEL_PATH = r"/root/miniconda3/envs/swj/src/output/shanghai_mu_t/mu_t_shanghai_model.pkl"

# 2) 上海筛选后的火灾数据（由筛选脚本生成）
DATA_PATH = r"D:\1a_Song_Clever\小论文\London\output\filtered_fire_incidents.csv"

# 3) 时间列列名
TIME_COLUMN_NAME = "lasj"

# 4) 可选：保存拟合出来的 α, β 参数的路径
#PARAM_OUTPUT_PATH = r"/root/miniconda3/envs/swj/src/output/hawkes_shanghai/hawkes_params_shanghai.pkl"


# -------------------------------------------------------------------
# 工具函数
# -------------------------------------------------------------------
def get_event_times_from_df(df, start_timestamp, total_duration_hours):
    """把 DataFrame 中的时间转为相对小时数"""
    print("\n--- 步骤: 对齐火灾事件时间 ---")
    event_times_numeric = (df.index - start_timestamp).total_seconds() / 3600.0
    mask = (event_times_numeric >= 0.0) & (event_times_numeric <= total_duration_hours)
    event_times_filtered = event_times_numeric[mask].values
    
    print(f"有效事件时间对齐完成: 共 {len(event_times_filtered)} 起参与拟合")
    return event_times_filtered


def calculate_negative_log_likelihood(params, event_times, mu_t_spline, T_max):
    """
    针对 Hawkes 过程：
        λ(t) = μ(t) + α * Σ_{t_i < t} exp(-β (t - t_i))
    的负对数似然函数 (Negative Log-Likelihood)。
    简化这个函数

    event_times: numpy 数组，事件发生时间（单位: 小时，相对于 start_timestamp）
    mu_t_spline: 连续背景 μ(t)，支持 mu_t_spline(t)
    T_max: 观测窗口长度（小时）
    """
    alpha, beta = params #params 是一个长度为 2 的数组或列表

    # 判断合法
    if alpha <= 0 or beta <= 0:
        return 1e20

    N = len(event_times) #计算事件总数
    if N == 0: #判断合法
        return 1e20

    # ----------------------------
    # 1) 计算 Σ log λ(t_i)
    # ----------------------------
    log_sum = 0.0
    A_i = 0.0  # 用于递推 Σ exp(-β (t_i - t_j))

    for i in range(N):
        t_i = event_times[i]
        if i > 0:
            dt = t_i - event_times[i - 1]
            # 递推公式: A_i = exp(-β Δt) * (A_{i-1} + 1)
            A_i = np.exp(-beta * dt) * (A_i + 1.0)

        mu_i = float(mu_t_spline(t_i))
        if mu_i < 0:
            # 理论上 μ(t) 应该非负，这里若有数值误差出现负数，则截断为 0
            mu_i = 0.0

        lambda_i = mu_i + alpha * A_i
        if lambda_i <= 0:
            # 避免 log(0) 或 log(负数)
            return 1e20

        log_sum += np.log(lambda_i)

    # ----------------------------
    # 2) 计算 ∫ λ(t) dt = ∫ μ(t) dt + ∫ excitation dt
    # ----------------------------

    # 背景项 ∫ μ(t) dt，用 CubicSpline 的 integrate 方法
    try:
        integral_mu = float(mu_t_spline.integrate(0.0, T_max))
    except Exception as e:
        print(f"错误: 计算 ∫ μ(t) dt 时失败: {e}")
        return 1e20

    # 自激部分积分 (有解析解):
    # ∑_i ∫_{t_i}^{T_max} α exp(-β (t - t_i)) dt
    #  = ∑_i α/β (1 - exp(-β (T_max - t_i)))
    dt_T = T_max - event_times
    dt_T = np.maximum(dt_T, 0.0)  # 防止数值误差出现负数
    integral_excitation = np.sum((alpha / beta) * (1.0 - np.exp(-beta * dt_T)))

    total_integral = integral_mu + integral_excitation

    log_likelihood = log_sum - total_integral
    neg_log_likelihood = -log_likelihood

    return neg_log_likelihood


def estimate_hawkes_parameters(event_times, mu_t_spline, T_max):
    """
    使用 scipy.optimize.minimize 对 Hawkes(α, β) 参数做 MLE 拟合。
    L-BFGS-B 算法
    """
    print("\n--- 开始进行 Hawkes 参数 (α, β) 的最大似然估计 ---")

    # 初始猜测值，可以根据经验调整
    initial_guess = np.array([0.5, 2.0])  # [alpha, beta]

    # 参数边界：α > 0, β > 0
    bounds = [(1e-6, None), (1e-6, None)]

    #使用L-BFGS-B算法
    result = minimize(
        calculate_negative_log_likelihood,
        initial_guess,
        args=(event_times, mu_t_spline, T_max),
        method="L-BFGS-B",
        bounds=bounds,
        options={"disp": True, "maxiter": 200}
    )

    if not result.success:
        print(f"⚠️ 警告: 优化未完全收敛。message = {result.message}")

    optimal_alpha, optimal_beta = result.x
    final_nll = result.fun

    print("\n✅ 拟合完成：")
    print(f"  最优 α (HAWKES_ALPHA) = {optimal_alpha:.6f}")
    print(f"  最优 β (HAWKES_BETA)  = {optimal_beta:.6f}")
    print(f"  最终负对数似然值 NLL   = {final_nll:.6f}")

    return optimal_alpha, optimal_beta, final_nll, result


if __name__ == "__main__":
    #测试运行
    # 1. 自动执行上一个脚本，直接从内存拿参数
    df, start_timestamp, total_duration_hours, mu_t_spline = get_mu_t_parameters()
    
    # 2. 对齐时间
    event_times = get_event_times_from_df(df, start_timestamp, total_duration_hours)
    
    # 3. 拟合参数
    alpha, beta, nll, result = estimate_hawkes_parameters(event_times, mu_t_spline, total_duration_hours)
    
    # 4. 输出最终结果
    print("\n========================================")
    print("🎉 霍克斯模型拟合完毕！最终结果：")
    print(f"  最优激发强度 α (Alpha) = {alpha:.6f}")
    print(f"  最优衰减速率 β (Beta)  = {beta:.6f}")
    print(f"  最终负对数似然值 NLL    = {nll:.6f}")
    print("========================================")