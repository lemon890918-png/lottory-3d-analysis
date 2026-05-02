#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中国福利彩票3D 遗漏值间隔分布检验
验证随机性：实际间隔分布 vs 理论几何分布(p=0.1)
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 路径
DATA_FILE = '/Users/wenxin/work/lottory-3d-analysis/data/fc3d_history.csv'
REPORT_DIR = '/Users/wenxin/work/lottory-3d-analysis/reports/prng_analysis'

os.makedirs(REPORT_DIR, exist_ok=True)

print("=" * 70)
print("中国福利彩票3D 遗漏值间隔分布检验 - 验证随机性")
print("=" * 70)

# 1. 读取数据
print("\n[1] 读取数据...")
df = pd.read_csv(DATA_FILE)
# 重置索引为0开始的顺序位置(不是issue号)
df = df.reset_index(drop=True)

print(f"    数据总期数: {len(df)}")
print(f"    数据范围: {df['issue'].iloc[0]} ~ {df['issue'].iloc[-1]}")
print(f"    列名: {list(df.columns)}")

# 确保数据类型
for col in ['hundred', 'ten', 'one']:
    df[col] = pd.to_numeric(df[col], errors='coerce').astype(int)

# 2. 计算遗漏间隔序列
def compute_gaps(seq_positions):
    """计算遗漏间隔序列
    seq_positions: 某个数字出现的顺序位置列表(0开始的整数)
    返回: 连续两次出现之间的期数间隔列表
    """
    if len(seq_positions) < 2:
        return np.array([])
    gaps = np.diff(seq_positions)
    return gaps

# 存储所有间隔数据
gap_data = {}
positions = ['hundred', 'ten', 'one']
position_names = {'hundred': '百位', 'ten': '十位', 'one': '个位'}

# 构建每个位置每个数字的出现顺序位置
for pos in positions:
    gap_data[pos] = {}
    for digit in range(10):
        # 找出该数字出现的位置索引(0开始的顺序位置)
        mask = df[pos] == digit
        seq_positions = np.where(mask)[0]  # 这就是0开始的顺序位置
        gaps = compute_gaps(seq_positions)
        gap_data[pos][digit] = gaps

print("\n[2] 遗漏间隔统计 (理论均值=10)")
print("-" * 70)

# KS检验结果汇总
ks_results = []

for pos in positions:
    print(f"\n>>> {position_names[pos]} <<<")
    for digit in range(10):
        gaps = gap_data[pos][digit]
        if len(gaps) == 0:
            continue
        
        mean_gap = np.mean(gaps)
        max_gap = np.max(gaps)
        count_long_gap = np.sum(gaps >= 20)  # 超长遗漏(≥20期)
        pct_long = count_long_gap / len(gaps) * 100
        
        # KS检验：实际分布 vs 理论几何分布(p=0.1)
        # 几何分布: P(X=k) = (1-p)^(k-1) * p, k=1,2,3,...
        # 理论均值 = 1/p = 10
        # 间隔从1开始，用 scipy.stats.geom(p=0.1, loc=0)
        ks_stat, ks_pvalue = stats.kstest(gaps, 'geom', args=(0.1,))
        
        ks_results.append({
            'position': pos,
            'position_name': position_names[pos],
            'digit': digit,
            'n_gaps': len(gaps),
            'mean_gap': mean_gap,
            'std_gap': np.std(gaps),
            'max_gap': max_gap,
            'pct_long': pct_long,
            'ks_stat': ks_stat,
            'ks_pvalue': ks_pvalue
        })
        
        flag = "**" if ks_pvalue < 0.05 else ""
        print(f"  数字{digit}: n={len(gaps):4d}, 均值={mean_gap:5.2f}, 最大={max_gap:4d}, "
              f"≥20期占比={pct_long:5.2f}%, KS_p={ks_pvalue:.4f} {flag}")

# 转为DataFrame便于分析
ks_df = pd.DataFrame(ks_results)

# 3. 整体分析
print("\n" + "=" * 70)
print("[3] 整体分析")
print("-" * 70)

# 偏离显著的数字(p<0.05)
sig_deviations = ks_df[ks_df['ks_pvalue'] < 0.05]
print(f"\n偏离几何分布显著的数字(p<0.05): {len(sig_deviations)}/30")
if len(sig_deviations) > 0:
    for _, row in sig_deviations.iterrows():
        print(f"  {row['position_name']} 数字{row['digit']}: "
              f"KS_p={row['ks_pvalue']:.4f}, 实际均值={row['mean_gap']:.2f}, 理论均值=10.00")

# 超长遗漏比例分析
print(f"\n超长遗漏(≥20期不出)比例分析:")
print(f"  理论值约10% (几何分布性质)")
overall_long_pct = ks_df['pct_long'].mean()
print(f"  实际平均值: {overall_long_pct:.2f}%")
print(f"  范围: {ks_df['pct_long'].min():.2f}% ~ {ks_df['pct_long'].max():.2f}%")

# 4. 绘制间隔分布对比图 - 每个位置数字组合
print("\n[4] 绘制间隔分布对比图...")

for pos in positions:
    for digit in range(10):
        gaps = gap_data[pos][digit]
        if len(gaps) < 10:
            continue
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # 实际分布直方图
        max_val = min(int(np.max(gaps)), 60)
        bins = range(1, max_val + 2)
        hist_vals, _ = np.histogram(gaps, bins=bins)
        hist_vals = hist_vals / len(gaps)  # 归一化
        
        bar_x = np.arange(1, len(hist_vals) + 1)
        ax.bar(bar_x, hist_vals, alpha=0.7, color='steelblue', label='实际分布', width=0.8)
        
        # 理论几何分布曲线
        p = 0.1
        theoretical = (1 - p) ** (bar_x - 1) * p
        ax.plot(bar_x, theoretical, 'ro-', markersize=4, label='理论几何分布(p=0.1)', linewidth=1.5)
        
        # 获取KS结果
        row = ks_df[(ks_df['position'] == pos) & (ks_df['digit'] == digit)].iloc[0]
        
        ax.set_xlabel('遗漏间隔(期数)', fontsize=11)
        ax.set_ylabel('概率', fontsize=11)
        ax.set_title(f'{position_names[pos]} 数字{digit} 遗漏间隔分布\n'
                     f'n={len(gaps)}, 均值={row["mean_gap"]:.2f}, '
                     f'KS_p={row["ks_pvalue"]:.4f}', fontsize=12)
        ax.legend(loc='upper right')
        ax.set_xlim(0, min(max_val, 45))
        ax.grid(True, alpha=0.3)
        
        fname = f'{pos}_gap_{digit}.png'
        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_DIR, fname), dpi=100)
        plt.close()

print(f"  已保存30张间隔分布图到 {REPORT_DIR}/")

# 5. 异常遗漏检测：找出历史上遗漏最长的20个记录
print("\n[5] 历史上遗漏最长的20个记录")
print("-" * 70)

# 重新构建每个位置-数字的出现顺序位置列表
all_occurrences = {}
for pos in positions:
    all_occurrences[pos] = {}
    for digit in range(10):
        mask = df[pos] == digit
        seq_positions = np.where(mask)[0]
        all_occurrences[pos][digit] = seq_positions

# 找出每个位置-数字组合的最大遗漏
max_gap_records = []

for pos in positions:
    for digit in range(10):
        seq_list = all_occurrences[pos][digit]
        if len(seq_list) < 2:
            continue
        
        # 计算所有间隔
        gaps = np.diff(seq_list)
        max_gap = np.max(gaps)
        max_idx = np.argmax(gaps)  # 在seq_list中的索引
        
        # 对应的期号
        start_seq = seq_list[max_idx]
        end_seq = seq_list[max_idx + 1]
        before_issue = df.loc[start_seq, 'issue']
        after_issue = df.loc[end_seq, 'issue']
        
        max_gap_records.append({
            'position': position_names[pos],
            'position_key': pos,
            'digit': digit,
            'gap': int(max_gap),
            'before_seq': int(start_seq),
            'after_seq': int(end_seq),
            'before_issue': before_issue,
            'after_issue': after_issue
        })

max_gap_df = pd.DataFrame(max_gap_records)
max_gap_df = max_gap_df.sort_values('gap', ascending=False).head(20)

print(f"\n{'排名':>4} {'位置':>4} {'数字':>4} {'遗漏期数':>8} {'出号前最后一期':>14} {'出号后第一期':>14}")
print("-" * 65)
for i, (_, row) in enumerate(max_gap_df.iterrows(), 1):
    print(f"{i:>4} {row['position']:>4} {row['digit']:>4} {row['gap']:>8} "
          f"{row['before_issue']:>14} {row['after_issue']:>14}")

# 保存最大遗漏记录
max_gap_df.to_csv(os.path.join(REPORT_DIR, 'max_gap_records.csv'), index=False, encoding='utf-8-sig')

# 6. 绘制遗漏最长Top20柱状图
print("\n[6] 绘制遗漏最长Top20柱状图...")

fig, ax = plt.subplots(figsize=(12, 7))
top20 = max_gap_df.head(20).reset_index(drop=True)
labels = [f"{row['position'][0]}{row['digit']}" for _, row in top20.iterrows()]
colors = ['crimson' if row['gap'] >= 30 else 'steelblue' for _, row in top20.iterrows()]
bars = ax.bar(range(20), top20['gap'], color=colors, alpha=0.8)

for i, (bar, gap) in enumerate(zip(bars, top20['gap'])):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{gap}', ha='center', va='bottom', fontsize=9)

ax.set_xticks(range(20))
ax.set_xticklabels(labels, rotation=0, fontsize=10)
ax.set_xlabel('位置+数字', fontsize=12)
ax.set_ylabel('最大遗漏期数', fontsize=12)
ax.set_title('历史上遗漏最长的20个位置-数字组合', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='crimson', alpha=0.8, label='≥30期(异常)'),
                   Patch(facecolor='steelblue', alpha=0.8, label='<30期')]
ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, 'max_gap_top20.png'), dpi=120)
plt.close()

# 7. 汇总KS检验p值热力图
print("\n[7] 绘制KS检验p值热力图...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('各位置数字遗漏间隔KS检验p值热力图 (p<0.05表示偏离几何分布)', fontsize=14)

pvalue_matrix = np.full((10, 3), np.nan)
pos_list = ['hundred', 'ten', 'one']
pos_idx = {p: i for i, p in enumerate(pos_list)}

for _, row in ks_df.iterrows():
    d = int(row['digit'])
    p = row['position']
    pvalue_matrix[d, pos_idx[p]] = row['ks_pvalue']

for i, pos in enumerate(pos_list):
    ax = axes[i]
    pvals = pvalue_matrix[:, i:i+1]
    im = ax.imshow(pvals, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    
    ax.set_xticks([0])
    ax.set_xticklabels([position_names[pos]])
    ax.set_yticks(range(10))
    ax.set_yticklabels(range(10))
    ax.set_title(f'{position_names[pos]}', fontsize=12)
    
    for d in range(10):
        val = pvals[d, 0]
        if not np.isnan(val):
            color = 'white' if val < 0.3 else 'black'
            ax.text(0, d, f'{val:.2f}', ha='center', va='center', 
                   fontsize=9, color=color, fontweight='bold')

fig.colorbar(im, ax=axes, shrink=0.6, label='KS检验p值')
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, 'ks_pvalue_heatmap.png'), dpi=120)
plt.close()

# 8. 绘制均值对比图(30个位置-数字组合)
print("\n[8] 绘制均值对比图...")

fig, ax = plt.subplots(figsize=(14, 6))
x_labels = []
means = []
colors = []

for pos in pos_list:
    for d in range(10):
        x_labels.append(f"{position_names[pos][0]}{d}")
        row = ks_df[(ks_df['position'] == pos) & (ks_df['digit'] == d)].iloc[0]
        means.append(row['mean_gap'])
        # 颜色: KS显著为红, 不显著为蓝
        colors.append('crimson' if row['ks_pvalue'] < 0.05 else 'steelblue')

bars = ax.bar(range(30), means, color=colors, alpha=0.8)
ax.axhline(y=10, color='green', linestyle='--', linewidth=2, label='理论均值=10')
ax.axhline(y=10+3, color='orange', linestyle=':', linewidth=1.5, label='理论均值+3σ(约13.7)')
ax.axhline(y=10-3, color='orange', linestyle=':', linewidth=1.5, label='理论均值-3σ(约6.3)')

for i, (bar, mean) in enumerate(zip(bars, means)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{mean:.1f}', ha='center', va='bottom', fontsize=7, rotation=45)

ax.set_xticks(range(30))
ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('平均遗漏间隔', fontsize=11)
ax.set_title('各位置-数字组合的平均遗漏间隔 vs 理论均值10', fontsize=13)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='crimson', alpha=0.8, label='KS检验p<0.05(偏离)'),
                   Patch(facecolor='steelblue', alpha=0.8, label='KS检验p≥0.05(正常)')]
ax.legend(handles=legend_elements + [plt.Line2D([0], [0], color='green', linestyle='--', label='理论均值=10')],
          loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, 'mean_gap_comparison.png'), dpi=120)
plt.close()

# 9. 结论
print("\n" + "=" * 70)
print("[9] 结论")
print("-" * 70)

total_tests = len(ks_df)
sig_count = len(ks_df[ks_df['ks_pvalue'] < 0.05])
expected_false_positives = total_tests * 0.05

print(f"\nKS检验结果汇总:")
print(f"  总检验数: {total_tests}")
print(f"  显著偏离(p<0.05): {sig_count}个 ({sig_count/total_tests*100:.1f}%)")
print(f"  期望假阳性数: ~{expected_false_positives:.1f}个 (5%基准)")

if sig_count > expected_false_positives * 2:
    verdict = "偏离显著"
    detail = "偏离比例远超5%假阳性基准，可能存在系统性偏离"
elif sig_count > expected_false_positives:
    verdict = "轻度偏离"
    detail = "偏离略多于预期，不排除随机波动"
else:
    verdict = "基本符合"
    detail = "偏离比例在随机波动范围内"

print(f"\n随机性判定: {verdict}")
print(f"  {detail}")

long_gap_overall = ks_df['pct_long'].mean()
print(f"\n超长遗漏(≥20期)比例:")
print(f"  理论值: 10.0%")
print(f"  实际均值: {long_gap_overall:.2f}%")
if long_gap_overall > 15:
    long_gap_conclusion = "明显偏高，存在聚集现象"
elif long_gap_overall > 12:
    long_gap_conclusion = "偏高，可能存在轻度聚集"
elif long_gap_overall < 5:
    long_gap_conclusion = "偏低，分布比理论更均匀"
else:
    long_gap_conclusion = "基本正常"
print(f"  结论: {long_gap_conclusion}")

# 均值偏离分析
mean_deviation = np.abs(ks_df['mean_gap'] - 10).mean()
print(f"\n均值偏离分析:")
print(f"  各组合实际均值与理论值10的平均绝对偏差: {mean_deviation:.2f}")
print(f"  理论标准误差: √(p/(1-p))/√n ≈ 9.5/√n (n为间隔数量)")

# 保存KS结果
ks_df.to_csv(os.path.join(REPORT_DIR, 'ks_test_results.csv'), index=False, encoding='utf-8-sig')

print("\n" + "=" * 70)
print("分析完成！")
print(f"报告目录: {REPORT_DIR}")
print("=" * 70)
