import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # (新增) 为模拟图表导入
import seaborn as sns
from datetime import datetime
import sys
import os
# --- 导入数据库连接引擎 ---
from sqlalchemy import create_engine

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def apply_boxed_inward_ticks(ax):
    """Use a four-sided frame and inward ticks for publication-style axes."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")

    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        length=4,
        width=1.0,
    )


def temporal_analysis(df, time_column):
    """时序分析"""
    print("\n📊 开始时序分析 (原始图表)...")
    print(f"  处理 {len(df):,} 条事件记录")

    df_temp = df.copy()
    print("⏰ 转换时间格式...")

    try:
        df_temp[time_column] = pd.to_datetime(df_temp[time_column], errors='coerce')

        # 移除转换后为空的时间戳
        invalid_count = df_temp[time_column].isna().sum()
        if invalid_count > 0:
            print(f"   警告: {invalid_count} 行无法被解析为有效日期时间，已被忽略。")
            df_temp = df_temp[df_temp[time_column].notna()]

        if df_temp.empty:
            print("❌ 错误: 没有有效的日期时间数据可供分析。")
            return None, None

    except Exception as e:
        print(f"   ⚠️  时间转换失败: {e}")
        print(f"   请检查 '{time_column}' 列的时间格式。")
        return None, None

    print("📅 提取时间特征...")
    df_temp['year'] = df_temp[time_column].dt.year
    df_temp['month'] = df_temp[time_column].dt.month
    df_temp['day'] = df_temp[time_column].dt.day
    df_temp['hour'] = df_temp[time_column].dt.hour
    df_temp['weekday'] = df_temp[time_column].dt.dayofweek
    df_temp['date'] = df_temp[time_column].dt.date

    # 时间戳设为索引
    df_temp = df_temp.set_index(time_column).sort_index()

    print("📈 计算各时间维度统计...")
    analysis_results = {}
    analysis_results['yearly'] = df_temp.groupby('year').size()
    analysis_results['monthly'] = df_temp.groupby('month').size()
    analysis_results['daily'] = df_temp.groupby('date').size()
    analysis_results['hourly'] = df_temp.groupby('hour').size()
    analysis_results['weekday'] = df_temp.groupby('weekday').size()
    analysis_results['year_month'] = df_temp.groupby(['year', 'month']).size()

    print("✅ 时序分析完成")
    if not analysis_results['daily'].empty:
        print(f"   分析了 {len(analysis_results['yearly'])} 年的数据")
        print(f"   覆盖 {len(analysis_results['daily'])} 天")
        print(f"   平均每日事件: {analysis_results['daily'].mean():.1f} 次")

    return df_temp, analysis_results


# --- 功能 3：时序可视化 ---
def create_temporal_visualizations(df, analysis_results, charts_dir):
    """生成时序分析可视化"""
    print("🎨 生成时序分析图表...")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    #suptitle_size = 20
    title_size = 20
    label_size = 17
    tick_label_size = 16
    annotation_size = 13

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    # (使用英文标题以避免字体问题)
    #fig.suptitle('Incident Temporal Analysis (Original)', fontsize=suptitle_size, fontweight='bold')
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    # 1. 年度趋势
    if len(analysis_results['yearly']) > 0:
        axes[0, 0].plot(analysis_results['yearly'].index,
                        analysis_results['yearly'].values,
                        marker='o', linewidth=2, markersize=6, color='steelblue')
        axes[0, 0].set_title('Annual Distribution', fontsize=title_size, fontweight='bold')
        axes[0, 0].set_xlabel('Year', fontsize=label_size)
        axes[0, 0].set_ylabel('Incident Count', fontsize=label_size)
        axes[0, 0].grid(True, alpha=0.3)

    # 2. 月度分布
    if len(analysis_results['monthly']) > 0:
        bars = axes[0, 1].bar(analysis_results['monthly'].index,
                              analysis_results['monthly'].values,
                              color='coral', alpha=0.7)
        axes[0, 1].set_title('Monthly Distribution', fontsize=title_size, fontweight='bold')
        axes[0, 1].set_xlabel('Month', fontsize=label_size)
        axes[0, 1].set_ylabel('Incident Count', fontsize=label_size)
        axes[0, 1].set_xticks(range(1, 13))
        axes[0, 1].set_xticklabels(month_names, rotation=45, fontsize=tick_label_size)
        axes[0, 1].grid(True, alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width() / 2., height,
                            f'{int(height)}', ha='center', va='bottom', fontsize=annotation_size)

    # 3. 小时分布
    if len(analysis_results['hourly']) > 0:
        bars = axes[1, 0].bar(analysis_results['hourly'].index,
                              analysis_results['hourly'].values,
                              color='lightblue', alpha=0.7)
        axes[1, 0].set_title('Hourly Distribution', fontsize=title_size, fontweight='bold')
        axes[1, 0].set_xlabel('Hour', fontsize=label_size)
        axes[1, 0].set_ylabel('Incident Count', fontsize=label_size)
        axes[1, 0].grid(True, alpha=0.3)

    # 4. 星期分布
    if len(analysis_results['weekday']) > 0:
        bars = axes[1, 1].bar(analysis_results['weekday'].index,
                              analysis_results['weekday'].values,
                              color='lightgreen', alpha=0.7)
        axes[1, 1].set_title('Weekday Distribution', fontsize=title_size, fontweight='bold')
        axes[1, 1].set_xlabel('Weekday', fontsize=label_size)
        axes[1, 1].set_ylabel('Incident Count', fontsize=label_size)
        min_y = analysis_results['weekday'].min()
        max_y = analysis_results['weekday'].max()
        padding = (max_y - min_y) * 0.1
        axes[1, 1].set_ylim(max(0, min_y - padding), max_y + padding)
        axes[1, 1].set_xticks(range(7))
        axes[1, 1].set_xticklabels(weekday_names, fontsize=tick_label_size)
        axes[1, 1].grid(True, alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2., height,
                            f'{int(height)}', ha='center', va='bottom', fontsize=annotation_size)

    # 5. 日度时间序列（全量绘制）
    # if len(analysis_results['daily']) > 0:
    #     daily_data = analysis_results['daily'].reset_index()
    #     daily_data.columns = ['date', 'count']
    #     daily_data['date'] = pd.to_datetime(daily_data['date'])
    #     daily_data = daily_data.sort_values('date')
    #
    #     axes[2, 0].plot(daily_data['date'], daily_data['count'],
    #                     linewidth=1, alpha=0.7, color='red')
    #     axes[2, 0].set_title('Daily Trend', fontsize=12, fontweight='bold')
    #     axes[2, 0].set_xlabel('Date')
    #     axes[2, 0].set_ylabel('Incident Count')
    #     axes[2, 0].tick_params(axis='x', rotation=45)
    #     axes[2, 0].grid(True, alpha=0.3)
    # else:
    #     axes[2, 0].text(0.5, 0.5, 'No daily data to display',
    #                     ha='center', va='center', transform=axes[2, 0].transAxes)

    # 6. 年月热力图
    # if len(analysis_results['year_month']) > 12:
    #     pivot_data = analysis_results['year_month'].reset_index()
    #     pivot_data.columns = ['year', 'month', 'count']
    #     pivot_table = pivot_data.pivot(index='year', columns='month', values='count')
    #     pivot_table = pivot_table.fillna(0)
    #
    #     sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='YlOrRd',
    #                 ax=axes[2, 1], cbar_kws={'label': 'Incident Count'})
    #     axes[2, 1].set_title('Year-Month Heatmap', fontsize=12, fontweight='bold')
    #     axes[2, 1].set_xlabel('Month')
    #     axes[2, 1].set_ylabel('Year')
    # else:
    #     axes[2, 1].text(0.5, 0.5, 'Not enough data for heatmap',
    #                     ha='center', va='center', transform=axes[2, 1].transAxes)

    for ax in axes.ravel():
        apply_boxed_inward_ticks(ax)
        ax.tick_params(axis='both', labelsize=tick_label_size)

    plt.tight_layout()

    chart1_path = os.path.join(charts_dir, f'temporal_analysis_{timestamp}.png')
    chart1_pdf_path = os.path.join(charts_dir, f'temporal_analysis_{timestamp}.pdf')
    plt.savefig(chart1_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(chart1_pdf_path, bbox_inches='tight', facecolor='white')
    print(f"✅ (原始) 时序分析图表已保存: {chart1_path}")
    print(f"✅ (原始) 矢量图表已保存: {chart1_pdf_path}")

    # plt.show() # 在脚本模式下注释掉
    return chart1_path


def main():
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    charts_dir = os.path.join(OUTPUT_DIR, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    
    input_csv_path = os.path.join(OUTPUT_DIR, "filtered_fire_incidents.csv")
    
    time_column = 'lasj' 

    print(f"\n📖 正在读取清洗后的数据: {input_csv_path}")
    if not os.path.exists(input_csv_path):
        print(f"❌ 错误：找不到文件 {input_csv_path}。请先运行之前的筛选脚本。")
        return

    try:
        # 读取 CSV
        fire_df = pd.read_csv(input_csv_path)
        print(f"✅ 成功加载 {len(fire_df):,} 条火灾记录。")
        
        # --- 时序分析 + 可视化 ---
        print(f"\n📊 开始执行时序分析及绘图...")
        
        # 调用你定义的分析函数
        df_with_features, analysis_results = temporal_analysis(
            fire_df, 
            time_column
        )

        if df_with_features is None or analysis_results is None:
            print("❌ 因时序分析准备数据失败，分析终止。")
            return

        # 生成时序可视化图表
        chart_path1 = create_temporal_visualizations(
            df_with_features,
            analysis_results,
            charts_dir
        )
        
        # 4. 汇总输出结果
        print("\n" + "="*40)
        print("🎉 可视化任务完成！")
        print(f"📊 分析图表位置: {chart_path1}")
        print(f"📅 覆盖时间范围: {df_with_features.index.min()} 至 {df_with_features.index.max()}")
        print("="*40)

    except Exception as e:
        print(f"❌ 程序运行失败: {e}")
        import traceback
        traceback.print_exc() # 打印具体错误堆栈，方便排查

if __name__ == "__main__":
    main()
