import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pickle
from scipy.interpolate import CubicSpline

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# -------------------------------------------------------------------
# 1. 配置参数
# -------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "output", "filtered_fire_incidents.csv")
TIME_COLUMN_NAME = "lasj"

OUTPUT_DIR = os.path.join(BASE_DIR, "output", "mu_t")
# -------------------------------------------------------------------


def load_and_preprocess(filepath, time_col):
        '''
        数据预处理
        '''
        print(f"--- 步骤 1: 正在加载数据 '{filepath}'...")
        df = pd.read_csv(filepath)# 读取 CSV 文件为 DataFrame

        if time_col not in df.columns:# 检查时间列是否存在
            print(f"错误: 在CSV中未找到时间列 '{time_col}'。")
            sys.exit(1)

        dt_index = pd.to_datetime(df[time_col], errors="coerce")# 将时间字符串转换为 Pandas 的 datetime 对象，无效时间转换为 NaT
        df = df[dt_index.notna()].copy()# 剔除转换失败（NaT）的无效数据
        df.index = dt_index[dt_index.notna()]# 将时间列设置为 DataFrame 的索引（Index）
        df = df.sort_index()# 按照时间先后顺序对数据进行排序

        print(f"数据加载成功。总共 {len(df)} 个有效事件。")
        print(f"时间范围: {df.index.min()} 到 {df.index.max()}")
        return df


def fit_hybrid_model(df):
        """
        泊松模型拟合(不含随机参数)
        lambda(t) = lambda_base
              * f_year(year)
              * f_seasonal(month)
              * f_weekly(weekday, hour)
        """
        print("\n--- 步骤 2: 拟合混合模型")

        # 计算数据集跨越的总天数和总小时数
        total_days = (df.index.max().date() - df.index.min().date()).days + 1
        total_hours = total_days * 24

        # 计算全局基础发生率：平均每小时发生多少起事件（lambda_base）
        lambda_base = len(df) / total_hours
        print(f"  基础速率 lambda_base = {lambda_base:.6f}")

        df = df.copy()
        df["year"] = df.index.year #从时间索引中提取出“年份”，并把它作为一个新列（Column）添加到数据表中。
        df["month"] = df.index.month
        df["day_of_week"] = df.index.dayofweek
        df["hour"] = df.index.hour

    
        total_weeks = total_days / 7.0
        hourly_counts = df.groupby(["day_of_week", "hour"]).size()#统计“星期几的几点钟”发生了多少起事件
        avg_hourly = hourly_counts / total_weeks# 计算该周每小时的火灾次数

        mux = pd.MultiIndex.from_product([range(7), range(24)],names=["day_of_week", "hour"])#168个
        f_weekly_shape = (avg_hourly.reindex(mux, fill_value=0) / lambda_base).fillna(0)#该时段的平均发生率 / 全局基础发生率
        f_weekly_shape.name = "f_weekly"

    
        daily_counts = df.resample("D").size()#每天的发生次数
        global_avg_daily = daily_counts.mean()#平均每天的发生次数

        avg_daily_by_month = daily_counts.groupby(daily_counts.index.month).mean()#每个月的日平均发生次数
        f_seasonal_factors = (avg_daily_by_month / global_avg_daily).reindex(range(1, 13), fill_value=1.0)#某月的日均发生率 / 全局日均发生率

    
        avg_daily_by_year = daily_counts.groupby(daily_counts.index.year).mean()#每一年的日均发生率
        f_year_factors = avg_daily_by_year / global_avg_daily#某年的日均发生率 / 全局日均发生率
    #year_range = range(daily_counts.index.year.min(), daily_counts.index.year.max() + 1)# 补齐年份区间
    #f_year_factors = f_year_factors.reindex(year_range, fill_value=1.0)

        return lambda_base, f_weekly_shape, f_seasonal_factors, f_year_factors


def generate_mu_t(df, lambda_base, f_weekly_shape, f_seasonal_factors, f_year_factors):
      """构建 mu_t 三次样条插值 并返回所需的核心参数"""
      print("\n--- 步骤 3: 构造 μ(t) spline ---")
      start_ts = df.index.min().normalize()
      end_ts = df.index.max().normalize() + pd.Timedelta(days=1)
      time_index = pd.date_range(
          start=start_ts,
          end=end_ts - pd.Timedelta(hours=1),
          freq="h"
      )

      factors = pd.DataFrame(index=time_index)
      factors["year"] = time_index.year
      factors["month"] = time_index.month
      factors["day_of_week"] = time_index.dayofweek
      factors["hour"] = time_index.hour

      factors["year_factor"] = factors["year"].map(f_year_factors)
      factors["seasonal_factor"] = factors["month"].map(f_seasonal_factors)

      weekly_df = f_weekly_shape.reset_index().set_index(["day_of_week", "hour"])
      factors = factors.join(weekly_df, on=["day_of_week", "hour"])

    # 修复：正确映射每月随机因子
      all_months = time_index.to_period("M").unique()
      monthly_random_factor = pd.Series(
          np.random.lognormal(0.0, 0.1, len(all_months)),
          index=all_months,
          name="random_factor"
      )
      factors["year_month"] = factors.index.to_period("M")
      factors["random_factor"] = factors["year_month"].map(monthly_random_factor)

    # 计算连续泊松强度
      poisson_model = (
        lambda_base *
        factors["year_factor"] *
        factors["seasonal_factor"] *
        factors["f_weekly"] *
        factors["random_factor"]
      )

    # 提取所需参数
      t_numeric = (poisson_model.index - start_ts).total_seconds() / 3600.0
      total_duration_hours = (end_ts - start_ts).total_seconds() / 3600.0
      t_points = np.append(t_numeric, total_duration_hours)
      mu_values = np.append(poisson_model.values, poisson_model.values[-1])
      mu_t_spline = CubicSpline(t_points, mu_values, extrapolate=False)

    # 直接返回供 Hawkes 使用的三个参数
      return start_ts, total_duration_hours, mu_t_spline

def get_mu_t_parameters():
    """
    提供给外部脚本调用的主函数。
    一键执行数据处理、模型拟合，并返回构建霍克斯模型所需的三个参数。
    """
    df = load_and_preprocess(FILE_PATH, TIME_COLUMN_NAME)
    lambda_base, f_weekly_shape, f_seasonal_factors, f_year_factors = fit_hybrid_model(df)
    
    start_timestamp, total_duration_hours, mu_t_spline = generate_mu_t(
        df, lambda_base, f_weekly_shape, f_seasonal_factors, f_year_factors
    )
    
    print("\n🎉 完成：已成功生成并提取 start_timestamp, total_duration_hours, mu_t_spline")
    return df, start_timestamp, total_duration_hours, mu_t_spline

if __name__ == "__main__":
    # 测试运行
    df, start_ts, duration, spline = get_mu_t_parameters()
    print("\n[测试输出]")
    print(f"起点: {start_ts}")
    print(f"总时长: {duration} 小时")
    print(f"样条对象: {spline}")
