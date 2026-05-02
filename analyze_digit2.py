#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
中国福利彩票3D历史数据分析 - 重点排查"十位数字2偏少"异常
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from scipy import stats
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ==================== 数据加载 ====================
print("=" * 60)
print("中国福利彩票3D历史数据分析")
print("=" * 60)

df = pd.read_csv('/Users/wenxin/work/lottory-3d-analysis/data/fc3d_history.csv')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter

print(f"\n数据概况:")
print(f"  总记录数: {len(df)}")
print(f"  时间范围: {df['date'].min().date()} 至 {df['date'].max().date()}")
print(f"  理论频率 (每个数字): 10.0%")

# ==================== 基础统计 ====================
print("\n" + "=" * 60)
print("1. 基础频率统计")
print("=" * 60)

positions = ['hundred', 'ten', 'one']
pos_names = {'hundred': '百位', 'ten': '十位', 'one': '个位'}

for pos in positions:
    counts = df[pos].value_counts().sort_index()
    freqs = counts / len(df) * 100
    print(f"\n{pos_names[pos]} ({pos}) 数字频率分布:")
    for digit in range(10):
        freq = freqs.get(digit, 0)
        deviation = freq - 10.0
        marker = " ***异常***" if abs(deviation) > 0.5 else ""
        print(f"  数字{digit}: {freq:.2f}% (偏差{deviation:+.2f}%){marker}")

# ==================== 十位数字2专项分析 ====================
print("\n" + "=" * 60)
print("2. 十位数字2专项分析")
print("=" * 60)

ten_two_count = (df['ten'] == 2).sum()
ten_two_freq = ten_two_count / len(df) * 100
print(f"\n十位数字2出现次数: {ten_two_count}")
print(f"十位数字2出现频率: {ten_two_freq:.2f}%")
print(f"理论频率: 10.00%")
print(f"偏差: {ten_two_freq - 10.0:.2f}%")

# ==================== 时间段分析 ====================
print("\n" + "=" * 60)
print("3. 时间段分析 - 数字2在十位的出现频率")
print("=" * 60)

# 按年分段
periods = {
    '2002-2010': (2002, 2010),
    '2011-2020': (2011, 2020),
    '2021-2026': (2021, 2026)
}

print("\n按年分段分析:")
for period_name, (start_year, end_year) in periods.items():
    mask = (df['year'] >= start_year) & (df['year'] <= end_year)
    period_df = df[mask]
    count = (period_df['ten'] == 2).sum()
    total = len(period_df)
    freq = count / total * 100 if total > 0 else 0
    expected = total / 10
    deviation = freq - 10.0
    marker = " ***偏少***" if deviation < -0.5 else (" ***偏多***" if deviation > 0.5 else "")
    print(f"  {period_name}: 出现{count}次/共{total}期, 频率{freq:.2f}% (偏差{deviation:+.2f}%){marker}")

# 按年份详细分析
print("\n按年份详细分析:")
yearly_ten2 = df.groupby('year').apply(lambda x: (x['ten'] == 2).sum() / len(x) * 100)
print(f"{'年份':<8}{'出现次数':<10}{'总期数':<10}{'频率':<10}{'偏差':<10}{'状态'}")
print("-" * 50)
for year in sorted(df['year'].unique()):
    count = (df[df['year'] == year]['ten'] == 2).sum()
    total = len(df[df['year'] == year])
    freq = count / total * 100
    deviation = freq - 10.0
    if deviation < -1.0:
        status = "显著偏少"
    elif deviation < -0.5:
        status = "偏少"
    elif deviation > 1.0:
        status = "显著偏多"
    elif deviation > 0.5:
        status = "偏多"
    else:
        status = "正常"
    print(f"{year:<8}{count:<10}{total:<10}{freq:.2f}%{'':<5}{deviation:+.2f}%{'':<5}{status}")

# ==================== 月/季度分析 ====================
print("\n" + "=" * 60)
print("4. 月/季度季节性分析")
print("=" * 60)

print("\n按月份分析 (所有年份汇总):")
monthly_ten2 = df.groupby('month').apply(lambda x: (x['ten'] == 2).sum() / len(x) * 100)
print(f"{'月份':<8}{'出现次数':<10}{'总期数':<10}{'频率':<10}{'偏差':<10}")
print("-" * 50)
for month in range(1, 13):
    count = (df[df['month'] == month]['ten'] == 2).sum()
    total = len(df[df['month'] == month])
    freq = count / total * 100 if total > 0 else 0
    deviation = freq - 10.0
    marker = " ***" if abs(deviation) > 0.5 else ""
    print(f"{month:<8}{count:<10}{total:<10}{freq:.2f}%{'':<5}{deviation:+.2f}%{marker}")

print("\n按季度分析:")
quarterly_ten2 = df.groupby('quarter').apply(lambda x: (x['ten'] == 2).sum() / len(x) * 100)
for q in [1, 2, 3, 4]:
    count = (df[df['quarter'] == q]['ten'] == 2).sum()
    total = len(df[df['quarter'] == q])
    freq = count / total * 100 if total > 0 else 0
    deviation = freq - 10.0
    marker = " ***" if abs(deviation) > 0.5 else ""
    print(f"  Q{q}: 出现{count}次/共{total}期, 频率{freq:.2f}% (偏差{deviation:+.2f}%){marker}")

# ==================== 滑动窗口分析 ====================
print("\n" + "=" * 60)
print("5. 滑动窗口分析 - 数字2在十位的频率变化")
print("=" * 60)

def sliding_window_analysis(df, pos, digit, window_sizes=[100, 200, 500]):
    """滑动窗口分析某位置某数字的出现频率"""
    results = {}
    series = (df[pos] == digit).astype(int)
    
    for window in window_sizes:
        frequencies = []
        indices = []
        for i in range(len(series) - window + 1):
            freq = series.iloc[i:i+window].mean() * 100
            frequencies.append(freq)
            indices.append(i + window // 2)
        results[window] = (indices, frequencies)
    
    return results

ten2_sliding = sliding_window_analysis(df, 'ten', 2, [100, 200, 500])

for window, (indices, frequencies) in ten2_sliding.items():
    avg = np.mean(frequencies)
    min_freq = np.min(frequencies)
    max_freq = np.max(frequencies)
    below_10 = sum(1 for f in frequencies if f < 10.0)
    print(f"\n窗口大小={window}期:")
    print(f"  平均频率: {avg:.2f}%")
    print(f"  最低频率: {min_freq:.2f}%")
    print(f"  最高频率: {max_freq:.2f}%")
    print(f"  低于理论值10%的窗口数: {below_10}/{len(frequencies)} ({below_10/len(frequencies)*100:.1f}%)")

# ==================== 各位置各数字频率异常检测 ====================
print("\n" + "=" * 60)
print("6. 各位置各数字频率异常检测 (超出理论值±5%标记为异常)")
print("=" * 60)

print(f"\n理论值: 10.00%, 异常阈值: 偏离超过0.5% (±5%)")
print("\n异常组合列表:")
anomalies = []
for pos in positions:
    counts = df[pos].value_counts()
    for digit in range(10):
        count = counts.get(digit, 0)
        freq = count / len(df) * 100
        deviation = freq - 10.0
        if abs(deviation) > 0.5:
            anomalies.append((pos_names[pos], digit, count, freq, deviation))
            
anomalies.sort(key=lambda x: x[4])  # 按偏差排序
print(f"{'位置':<8}{'数字':<8}{'出现次数':<10}{'频率':<10}{'偏差':<10}")
print("-" * 50)
for pos_name, digit, count, freq, deviation in anomalies:
    direction = "偏少" if deviation < 0 else "偏多"
    print(f"{pos_name:<8}{digit:<8}{count:<10}{freq:.2f}%{'':<5}{deviation:+.2f}%  {direction}")

# ==================== 卡方检验 ====================
print("\n" + "=" * 60)
print("7. 卡方检验 - 找出p<0.1的异常组合")
print("=" * 60)

significant_results = []
for pos in positions:
    counts = df[pos].value_counts()
    observed = np.array([counts.get(d, 0) for d in range(10)])
    expected = np.array([len(df) / 10] * 10)
    
    chi2, p_value = stats.chisquare(observed, expected)
    
    print(f"\n{pos_names[pos]} 位置卡方检验:")
    print(f"  卡方值: {chi2:.4f}")
    print(f"  p值: {p_value:.6f}")
    
    # 找出每个数字的偏差
    for digit in range(10):
        obs = counts.get(digit, 0)
        exp = len(df) / 10
        chi2_single = (obs - exp) ** 2 / exp
        p_single = 1 - stats.chi2.cdf(chi2_single, df=1)
        
        if p_single < 0.1:
            deviation = (obs - exp) / exp * 100
            direction = "偏少" if obs < exp else "偏多"
            significant_results.append((pos_names[pos], digit, obs, exp, chi2_single, p_single, deviation, direction))
            print(f"    数字{digit}: 观测={obs:.0f}, 期望={exp:.0f}, p={p_single:.6f}, 偏差{deviation:+.2f}% ({direction})")

print("\n\np<0.1的异常组合汇总 (按p值排序):")
print(f"{'位置':<8}{'数字':<8}{'观测值':<10}{'期望值':<10}{'卡方':<10}{'p值':<12}{'偏差':<10}")
print("-" * 70)
significant_results.sort(key=lambda x: x[5])
for pos_name, digit, obs, exp, chi2_val, p_val, deviation, direction in significant_results:
    print(f"{pos_name:<8}{digit:<8}{obs:<10.0f}{exp:<10.1f}{chi2_val:<10.4f}{p_val:<12.6f}{deviation:+.2f}%  {direction}")

# ==================== 连续出现分析 ====================
print("\n" + "=" * 60)
print("8. 数字2在十位连续出现/连续不出现分析")
print("=" * 60)

ten_series = (df['ten'] == 2).values

# 连续出现
max_consecutive_appear = 0
current_appear = 0
for val in ten_series:
    if val:
        current_appear += 1
        max_consecutive_appear = max(max_consecutive_appear, current_appear)
    else:
        current_appear = 0

# 连续不出现
max_consecutive_absent = 0
current_absent = 0
for val in ten_series:
    if not val:
        current_absent += 1
        max_consecutive_absent = max(max_consecutive_absent, current_absent)
    else:
        current_absent = 0

print(f"\n数字2在十位连续出现:")
print(f"  最长连续出现期数: {max_consecutive_appear}")

# 找出最长连续出现的起止位置
current_count = 0
start_idx = 0
max_start = 0
max_count = 0
for i, val in enumerate(ten_series):
    if val:
        if current_count == 0:
            start_idx = i
        current_count += 1
        if current_count > max_count:
            max_count = current_count
            max_start = start_idx
    else:
        current_count = 0

if max_count > 0:
    start_issue = df.iloc[max_start]['issue']
    end_issue = df.iloc[max_start + max_count - 1]['issue']
    start_date = df.iloc[max_start]['date'].date()
    end_date = df.iloc[max_start + max_count - 1]['date'].date()
    print(f"  最长连续出现在: {start_issue}({start_date}) 至 {end_issue}({end_date})")

print(f"\n数字2在十位连续不出现:")
print(f"  最长连续不出现期数: {max_consecutive_absent}")

# 找出最长连续不出现的起止位置
current_count = 0
start_idx = 0
for i, val in enumerate(ten_series):
    if not val:
        if current_count == 0:
            start_idx = i
        current_count += 1
        if current_count > max_count:
            max_count = current_count
            max_start = start_idx
    else:
        current_count = 0

if max_count > 0:
    start_issue = df.iloc[max_start]['issue']
    end_issue = df.iloc[max_start + max_count - 1]['issue']
    start_date = df.iloc[max_start]['date'].date()
    end_date = df.iloc[max_start + max_count - 1]['date'].date()
    print(f"  最长连续不出现于: {start_issue}({start_date}) 至 {end_issue}({end_date})")

# ==================== 生成可视化图表 ====================
print("\n" + "=" * 60)
print("9. 生成可视化图表")
print("=" * 60)

output_dir = '/Users/wenxin/work/lottory-3d-analysis/reports/prng_analysis/'

# 图1: 十位数字2滑动窗口频率变化曲线
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
window_sizes = [100, 200, 500]

for idx, window in enumerate(window_sizes):
    indices, frequencies = ten2_sliding[window]
    ax = axes[idx]
    ax.plot(indices, frequencies, color=colors[idx], linewidth=0.8, alpha=0.8)
    ax.axhline(y=10.0, color='red', linestyle='--', linewidth=1.5, label='理论值 10%')
    ax.fill_between(indices, frequencies, 10.0, 
                     where=[f < 10.0 for f in frequencies], 
                     color='red', alpha=0.3, label='低于理论值')
    ax.set_title(f'十位数字2出现频率 (滑动窗口={window}期)', fontsize=12)
    ax.set_xlabel('期号索引')
    ax.set_ylabel('频率 (%)')
    ax.set_ylim(0, 20)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 标注最大和最小值
    min_idx = np.argmin(frequencies)
    ax.annotate(f'最低: {frequencies[min_idx]:.2f}%', 
                xy=(indices[min_idx], frequencies[min_idx]),
                xytext=(indices[min_idx]+50, frequencies[min_idx]+2),
                fontsize=8, color='red')

plt.tight_layout()
plt.savefig(output_dir + 'ten_digit2_sliding_window.png', dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: ten_digit2_sliding_window.png")

# 图2: 十位数字2按年份频率热力图
fig, ax = plt.subplots(figsize=(16, 8))
years = sorted(df['year'].unique())
heatmap_data = []

for year in years:
    year_df = df[df['year'] == year]
    counts = year_df['ten'].value_counts()
    freqs = [(counts.get(d, 0) / len(year_df) * 100) for d in range(10)]
    heatmap_data.append(freqs)

heatmap_data = np.array(heatmap_data)
im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=5, vmax=15)

# 添加数值标注
for i in range(len(years)):
    for j in range(10):
        color = 'white' if abs(heatmap_data[i][j] - 10) > 3 else 'black'
        ax.text(j, i, f'{heatmap_data[i][j]:.1f}', ha='center', va='center', fontsize=7, color=color)

ax.set_xticks(range(10))
ax.set_xticklabels([str(d) for d in range(10)])
ax.set_yticks(range(len(years)))
ax.set_yticklabels(years)
ax.set_xlabel('数字')
ax.set_ylabel('年份')
ax.set_title('十位数字频率热力图 (各年份)', fontsize=14)
plt.colorbar(im, ax=ax, label='频率 (%)')
plt.tight_layout()
plt.savefig(output_dir + 'ten_position_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: ten_position_heatmap.png")

# 图3: 各位置各数字频率对比 (柱状图)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
x = np.arange(10)

for idx, pos in enumerate(positions):
    counts = df[pos].value_counts().sort_index()
    freqs = [(counts.get(d, 0) / len(df) * 100) for d in range(10)]
    colors_bar = ['red' if abs(f - 10) > 0.5 else '#1f77b4' for f in freqs]
    
    axes[idx].bar(x, freqs, color=colors_bar, alpha=0.7, edgecolor='black')
    axes[idx].axhline(y=10.0, color='red', linestyle='--', linewidth=1.5, label='理论值 10%')
    axes[idx].set_xlabel('数字')
    axes[idx].set_ylabel('频率 (%)')
    axes[idx].set_title(f'{pos_names[pos]} 频率分布', fontsize=12)
    axes[idx].set_xticks(x)
    axes[idx].set_xticklabels([str(d) for d in range(10)])
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3, axis='y')
    
    # 标注异常值
    for j, f in enumerate(freqs):
        if abs(f - 10) > 0.5:
            axes[idx].annotate(f'{f:.1f}%', xy=(j, f), xytext=(j, f + 0.3),
                               ha='center', fontsize=8, color='red')

plt.tight_layout()
plt.savefig(output_dir + 'all_positions_frequency.png', dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: all_positions_frequency.png")

# 图4: 三位置综合热力图 (横轴=数字0-9, 纵轴=年份)
fig, axes = plt.subplots(1, 3, figsize=(18, 10))

for pos_idx, pos in enumerate(positions):
    ax = axes[pos_idx]
    heatmap_data = []
    
    for year in years:
        year_df = df[df['year'] == year]
        counts = year_df[pos].value_counts()
        freqs = [(counts.get(d, 0) / len(year_df) * 100) for d in range(10)]
        heatmap_data.append(freqs)
    
    heatmap_data = np.array(heatmap_data)
    im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=5, vmax=15)
    
    # 添加数值标注
    for i in range(len(years)):
        for j in range(10):
            color = 'white' if abs(heatmap_data[i][j] - 10) > 3 else 'black'
            ax.text(j, i, f'{heatmap_data[i][j]:.1f}', ha='center', va='center', fontsize=6, color=color)
    
    ax.set_xticks(range(10))
    ax.set_xticklabels([str(d) for d in range(10)])
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels(years, fontsize=7)
    ax.set_xlabel('数字')
    ax.set_ylabel('年份')
    ax.set_title(f'{pos_names[pos]} 位置频率热力图', fontsize=12)
    plt.colorbar(im, ax=ax, label='频率 (%)')

plt.tight_layout()
plt.savefig(output_dir + 'three_positions_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: three_positions_heatmap.png")

# 图5: 数字2在十位的年度频率趋势
fig, ax = plt.subplots(figsize=(14, 6))

yearly_ten2 = df.groupby('year').apply(lambda x: (x['ten'] == 2).sum() / len(x) * 100)
bars = ax.bar(yearly_ten2.index, yearly_ten2.values, color='steelblue', alpha=0.7, edgecolor='black')
ax.axhline(y=10.0, color='red', linestyle='--', linewidth=2, label='理论值 10%')

# 标注偏差大的年份
for i, (year, freq) in enumerate(yearly_ten2.items()):
    deviation = freq - 10.0
    if abs(deviation) > 0.5:
        color = 'red' if deviation < 0 else 'green'
        ax.annotate(f'{freq:.2f}%', xy=(year, freq), xytext=(year, freq + (0.5 if deviation > 0 else -0.8)),
                   ha='center', fontsize=8, color=color, fontweight='bold')

ax.set_xlabel('年份')
ax.set_ylabel('频率 (%)')
ax.set_title('十位数字2年度频率变化', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 15)
plt.tight_layout()
plt.savefig(output_dir + 'ten_digit2_yearly_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: ten_digit2_yearly_trend.png")

# 图6: 月份季节性分析
fig, ax = plt.subplots(figsize=(12, 5))
monthly_ten2 = df.groupby('month').apply(lambda x: (x['ten'] == 2).sum() / len(x) * 100)
bars = ax.bar(monthly_ten2.index, monthly_ten2.values, color='#2ca02c', alpha=0.7, edgecolor='black')
ax.axhline(y=10.0, color='red', linestyle='--', linewidth=2, label='理论值 10%')

for m, freq in monthly_ten2.items():
    deviation = freq - 10.0
    if abs(deviation) > 0.5:
        color = 'red' if deviation < 0 else 'green'
        ax.annotate(f'{freq:.2f}%', xy=(m, freq), xytext=(m, freq + (0.3 if deviation > 0 else -0.5)),
                   ha='center', fontsize=8, color=color)

ax.set_xlabel('月份')
ax.set_ylabel('频率 (%)')
ax.set_title('十位数字2月度频率分布 (季节性分析)', fontsize=14)
ax.set_xticks(range(1, 13))
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(output_dir + 'ten_digit2_monthly.png', dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: ten_digit2_monthly.png")

# 图7: 连续出现/不出现分析
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

# 连续出现分析
appear_streaks = []
absent_streaks = []
current_appear = 0
current_absent = 0

for val in ten_series:
    if val:
        if current_absent > 0:
            absent_streaks.append(current_absent)
            current_absent = 0
        current_appear += 1
    else:
        if current_appear > 0:
            appear_streaks.append(current_appear)
            current_appear = 0
        current_absent += 1

if current_appear > 0:
    appear_streaks.append(current_appear)
if current_absent > 0:
    absent_streaks.append(current_absent)

ax1.hist(appear_streaks, bins=range(1, max(appear_streaks)+2), color='green', alpha=0.7, edgecolor='black')
ax1.set_xlabel('连续出现期数')
ax1.set_ylabel('频次')
ax1.set_title('十位数字2连续出现期数分布', fontsize=12)
ax1.axvline(x=np.mean(appear_streaks), color='red', linestyle='--', label=f'均值: {np.mean(appear_streaks):.2f}')
ax1.legend()

ax2.hist(absent_streaks, bins=range(1, max(absent_streaks)+2), color='orange', alpha=0.7, edgecolor='black')
ax2.set_xlabel('连续不出现期数')
ax2.set_ylabel('频次')
ax2.set_title('十位数字2连续不出现期数分布', fontsize=12)
ax2.axvline(x=np.mean(absent_streaks), color='red', linestyle='--', label=f'均值: {np.mean(absent_streaks):.2f}')
ax2.legend()

plt.tight_layout()
plt.savefig(output_dir + 'ten_digit2_streak_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  已保存: ten_digit2_streak_analysis.png")

print("\n所有图表已保存到: " + output_dir)

# ==================== 结论总结 ====================
print("\n" + "=" * 60)
print("分析结论")
print("=" * 60)

print("\n1. 十位数字2的总体情况:")
print(f"   - 总体出现频率: {ten_two_freq:.2f}% (理论值10.00%)")
print(f"   - 偏差: {ten_two_freq - 10.0:.2f}%")

# 判断是否为全时段现象
early_period = df[df['year'] <= 2010]
mid_period = df[(df['year'] >= 2011) & (df['year'] <= 2020)]
late_period = df[df['year'] >= 2021]

early_freq = (early_period['ten'] == 2).sum() / len(early_period) * 100
mid_freq = (mid_period['ten'] == 2).sum() / len(mid_period) * 100
late_freq = (late_period['ten'] == 2).sum() / len(late_period) * 100

all_below = early_freq < 10.0 and mid_freq < 10.0 and late_freq < 10.0

print(f"\n2. 时间段分析:")
print(f"   - 2002-2010年: {early_freq:.2f}% {'偏少' if early_freq < 10 else '偏多'}")
print(f"   - 2011-2020年: {mid_freq:.2f}% {'偏少' if mid_freq < 10 else '偏多'}")
print(f"   - 2021-2026年: {late_freq:.2f}% {'偏少' if late_freq < 10 else '偏多'}")

if all_below:
    print(f"\n3. 结论: 数字2在十位偏少是【全时段现象】")
    print(f"   三个时期的频率都低于理论值10%，表明这不是某个特定时间段的异常，")
    print(f"   而是贯穿整个历史数据的一致性偏少现象。")
else:
    print(f"\n3. 结论: 数字2在十位偏少是【特定时间段异常】")
    if early_freq < 10:
        print(f"   主要异常出现在2002-2010年")
    if mid_freq < 10:
        print(f"   主要异常出现在2011-2020年")
    if late_freq < 10:
        print(f"   主要异常出现在2021-2026年")

print(f"\n4. 统计检验结果:")
if len(significant_results) > 0:
    print(f"   发现 {len(significant_results)} 个p<0.1的异常组合")
    ten2_significant = [r for r in significant_results if r[0] == '十位' and r[1] == 2]
    if ten2_significant:
        print(f"   其中包括: 十位数字2 (p={ten2_significant[0][5]:.6f})")
else:
    print(f"   未发现p<0.1的统计显著异常组合")

print("\n" + "=" * 60)
print("分析完成")
print("=" * 60)
