#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from ld_possion import get_mu_t_parameters
from ld_estimate_hawkes_parameters import get_event_times_from_df, estimate_hawkes_parameters

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# 输出目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output/ld_simulation")


# ----------------------------------------------------------
#  Hawkes 模拟（指数核递推 + thinning）
# ----------------------------------------------------------

def simulate_hawkes_fast(mu_t_spline, mu_global_max, T_max, alpha, beta):
    """
    λ(t) = μ(t) + α * A(t)
    A(t) = Σ_{t_i < t} exp(-β (t - t_i))

    递推：在任意时间推进 dt：A <- A * exp(-β dt)
          接受事件：A <- A + 1
    复杂度 O(N)
    """
    print("\n--- 开始 Hawkes 加速模拟 (O(N)) ---")
    print(f"  alpha={alpha:.6f}, beta={beta:.6f}")

    t = 0.0
    A = 0.0
    last_t = 0.0
    history = []

    pbar = tqdm(desc="Simulating Hawkes (accepted events)", unit="events")

    while t < T_max:
        # 安全上界（A 之后只会衰减，所以 mu_global_max + αA 是上界）
        M = mu_global_max + alpha * A
        if M <= 0:
            break

        # 采样候选间隔
        dt = np.random.exponential(1.0 / M)
        t = last_t + dt
        if t > T_max:
            break

        # 将 A 衰减到候选时刻 t
        A = A * np.exp(-beta * (t - last_t))

        # 真实强度 λ(t)
        mu_val = float(mu_t_spline(t))
        if mu_val < 0:
            mu_val = 0.0
        lam = mu_val + alpha * A
        if lam < 0:
            lam = 0.0

        # thinning
        if np.random.rand() <= lam / M:
            history.append(t)
            A += 1.0
            pbar.update(1)

        last_t = t

    pbar.close()
    history = np.array(history, dtype=float)
    print(f"✅ 模拟完成：生成事件数 = {len(history)}")
    return history


# ----------------------------------------------------------
# 画真实 vs 模拟对比图（长期日计数 + 90日均线）
# ----------------------------------------------------------

def plot_real_vs_sim(df_real, df_sim, outdir):
    print("\n--- [阶段 4] 正在生成真实 vs 模拟对比图 ---")

    real_daily = df_real.resample("D").size()
    sim_daily = df_sim.resample("D").size()

    real_ma = real_daily.rolling(90, center=True).mean() if len(real_daily) >= 90 else None
    sim_ma  = sim_daily.rolling(90, center=True).mean() if len(sim_daily) >= 90 else None

    plt.figure(figsize=(20, 12), dpi=150)

    # 真实数据图表
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(real_daily.index, real_daily.values, color="cornflowerblue", linewidth=1.0, alpha=0.55, label="Original")
    if real_ma is not None:
        ax1.plot(real_ma.index, real_ma.values, color="navy", linestyle="--", linewidth=2.0, alpha=0.95, label="90-Day MA")
    ax1.set_title("London - Original Daily Count", fontsize=16)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Events / Day")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left")

    # 模拟数据图表
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(sim_daily.index, sim_daily.values, color="lightcoral", linewidth=1.0, alpha=0.55, label="Simulated Hawkes")
    if sim_ma is not None:
        ax2.plot(sim_ma.index, sim_ma.values, color="darkred", linestyle="--", linewidth=2.0, alpha=0.95, label="90-Day MA")
    ax2.set_title("London - SECS Simulated Daily Count", fontsize=16)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Events / Day")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="upper right")

    # 统一 Y 轴范围
    if len(real_daily) and len(sim_daily):
        ymin = min(real_daily.min(), sim_daily.min())
        ymax = max(real_daily.max(), sim_daily.max())
        ax1.set_ylim(ymin, ymax)
        ax2.set_ylim(ymin, ymax)

    outpath = os.path.join(outdir, "London_Hawkes_LongTerm_Comparison_fast.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"✅ 对比图已保存：{outpath}")


def get_simulated_ts():
    """
    仅返回 sim_ts最终模拟时间
    用于后续调取前60个事件
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 阶段 1：获取参数
    print("\n[模拟] 初始化参数...")
    df_real, start_ts, T_max, mu_t_spline = get_mu_t_parameters()

    t_samples = np.arange(0, T_max, 1.0)
    mu_global_max = float(np.max(mu_t_spline(t_samples)))

    # 阶段 2：拟合参数
    print("[模拟] 计算 Hawkes 参数...")
    event_times = get_event_times_from_df(df_real, start_ts, T_max)
    alpha, beta, nll, result = estimate_hawkes_parameters(event_times, mu_t_spline, T_max)

    # 阶段 3：模拟
    print("[模拟] 开始 Hawkes 模拟...")
    sim_times = simulate_hawkes_fast(
        mu_t_spline=mu_t_spline,
        mu_global_max=mu_global_max,
        T_max=T_max,
        alpha=alpha,
        beta=beta
    )

    # 阶段 4：生成时间戳（只到这里，不保存、不画图）
    sim_ts = start_ts + pd.to_timedelta(sim_times, unit="h")
    sim_ts = sim_ts.view('int64') / 10**9
    
    print("[模拟] 完成，返回时间序列 sim_ts")
    return sim_ts


# ----------------------------------------------------------
# 主程序
# ----------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 【阶段 1】: 自动生成背景概率 (调用第一个脚本)
    print("\n=======================================================")
    print("  [阶段 1] 初始化背景模型与真实数据...")
    df_real, start_ts, T_max, mu_t_spline = get_mu_t_parameters()
    
    # 算一下峰值，供后面加速模拟用
    t_samples = np.arange(0, T_max, 1.0)
    mu_global_max = float(np.max(mu_t_spline(t_samples)))
    
    # 【阶段 2】: 自动拟合霍克斯参数 (调用第二个脚本)
    print("\n=======================================================")
    print("  [阶段 2] 动态计算 Hawkes 参数 (Alpha & Beta)...")
    event_times = get_event_times_from_df(df_real, start_ts, T_max)
    
    # 接收 4 个返回值，注意解包
    alpha, beta, nll, result = estimate_hawkes_parameters(event_times, mu_t_spline, T_max)

    # 【阶段 3】: 进行模拟
    print("\n=======================================================")
    sim_times = simulate_hawkes_fast(
        mu_t_spline=mu_t_spline,
        mu_global_max=mu_global_max,
        T_max=T_max,
        alpha=alpha,
        beta=beta
    )

    # 【阶段 4】: 保存结果并画图
    print("\n=======================================================")
    sim_ts = start_ts + pd.to_timedelta(sim_times, unit="h")
    df_sim = pd.DataFrame(index=pd.to_datetime(sim_ts))
    df_sim.index.name = "timestamp"

    sim_csv = os.path.join(OUTPUT_DIR, "simulated_london5.csv")
    df_sim.to_csv(sim_csv)
    print(f"  -> 模拟事件列表已导出: {sim_csv}")

    plot_real_vs_sim(df_real, df_sim, OUTPUT_DIR)

    print("\n🎉 恭喜！全自动流水线闭环：数据清洗 -> 背景概率 -> 霍克斯参数估计 -> 最终模拟预测 全部完成！")


if __name__ == "__main__":
    main()
