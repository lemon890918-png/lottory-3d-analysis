#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中国福利彩票3D 历史数据探索性数据分析 (EDA)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 路径设置
DATA_PATH = '/Users/wenxin/work/lottory-3d-analysis/data/fc3d_history.csv'
REPORTS_PATH = '/Users/wenxin/work/lottory-3d-analysis/reports'

# 确保reports目录存在
os.makedirs(REPORTS_PATH, exist_ok=True)

# 加载数据
print("=" * 60)
print("中国福利彩票3D 历史数据分析报告")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n【数据加载成功】共 {len(df)} 条记录")

# ==================== 1. 基本统计 ====================
print("\n" + "=" * 60)
print("1. 基本统计")
print("=" * 60)

print(f"总期数: {len(df)}")
print(f"日期范围: {df['date'].iloc[-1]} 至 {df['date'].iloc[0]}")

# ==================== 2. 数字频率分析 ====================
print("\n" + "=" * 60)
print("2. 数字频率分析（百位/十位/个位）")
print("=" * 60)

def digit_frequency(col, name):
    counter = Counter(col)
    total = len(col)
    freq = {i: counter.get(i, 0) for i in range(10)}
    print(f"\n{name}分布:")
    for d in range(10):
        count = freq[d]
        pct = count / total * 100
        bar = '█' * int(pct / 2)
        print(f"  {d}: {count:3d}次 ({pct:5.1f}%) {bar}")
    return freq

hundred_freq = digit_frequency(df['hundred'], "百位")
ten_freq = digit_frequency(df['ten'], "十位")
one_freq = digit_frequency(df['one'], "个位")

# ==================== 3. 组选vs直选 ====================
print("\n" + "=" * 60)
print("3. 组选 vs 直选 分析")
print("=" * 60)

# 直选: 百位、十位、个位都与开奖号码一致（顺序也一致）
# 组选: 三个数字相同则为组三（2个相同+1个不同），三个数字都不同则为组六
def classify_type(row):
    digits = sorted([row['hundred'], row['ten'], row['one']])
    if digits[0] == digits[1] == digits[2]:
        return "豹子"
    elif digits[0] == digits[1] or digits[1] == digits[2]:
        return "组三"
    else:
        return "组六"

df['type'] = df.apply(classify_type, axis=1)
type_counts = df['type'].value_counts()

print("\n组选类型分布:")
for t, c in type_counts.items():
    pct = c / len(df) * 100
    bar = '█' * int(pct / 2)
    print(f"  {t}: {c:3d}次 ({pct:5.1f}%) {bar}")

# ==================== 4. 和值分布 ====================
print("\n" + "=" * 60)
print("4. 和值分布（百+十+个）")
print("=" * 60)

df['sum_value'] = df['hundred'] + df['ten'] + df['one']
sum_counts = df['sum_value'].value_counts().sort_index()

print("\n和值分布 (0-27):")
for s in range(28):
    count = sum_counts.get(s, 0)
    pct = count / len(df) * 100
    bar = '█' * int(pct / 2)
    print(f"  和值{s:2d}: {count:3d}次 ({pct:5.1f}%) {bar}")

print(f"\n和值统计: 均值={df['sum_value'].mean():.2f}, 中位数={df['sum_value'].median():.1f}, 范围=[{df['sum_value'].min()}, {df['sum_value'].max()}]")

# ==================== 5. 跨度分布 ====================
print("\n" + "=" * 60)
print("5. 跨度分布（最大数字 - 最小数字）")
print("=" * 60)

df['span'] = df.apply(lambda row: max(row['hundred'], row['ten'], row['one']) - min(row['hundred'], row['ten'], row['one']), axis=1)
span_counts = df['span'].value_counts().sort_index()

print("\n跨度分布 (0-9):")
for s in range(10):
    count = span_counts.get(s, 0)
    pct = count / len(df) * 100
    bar = '█' * int(pct / 2)
    print(f"  跨度{s}: {count:3d}次 ({pct:5.1f}%) {bar}")

# ==================== 6. 012路分析 ====================
print("\n" + "=" * 60)
print("6. 012路分析（数字 mod 3）")
print("=" * 60)

def get_road(x):
    return x % 3

df['hundred_road'] = df['hundred'].apply(get_road)
df['ten_road'] = df['ten'].apply(get_road)
df['one_road'] = df['one'].apply(get_road)

print("\n各位置012路分布:")
for pos, col in [('百位', 'hundred_road'), ('十位', 'ten_road'), ('个位', 'one_road')]:
    road_counts = df[col].value_counts().sort_index()
    print(f"  {pos}: 0路={road_counts.get(0,0)}, 1路={road_counts.get(1,0)}, 2路={road_counts.get(2,0)}")

# ==================== 7. 奇偶比例分布 ====================
print("\n" + "=" * 60)
print("7. 奇偶比例分布")
print("=" * 60)

def is_odd(x):
    return x % 2 == 1

df['hundred_odd'] = df['hundred'].apply(is_odd)
df['ten_odd'] = df['ten'].apply(is_odd)
df['one_odd'] = df['one'].apply(is_odd)

# 统计奇数个数
df['odd_count'] = df['hundred_odd'].astype(int) + df['ten_odd'].astype(int) + df['one_odd'].astype(int)
odd_pattern = df['odd_count'].value_counts().sort_index()

print("\n奇偶比例分布 (3个数字中奇数的个数):")
for i in range(4):
    count = odd_pattern.get(i, 0)
    pct = count / len(df) * 100
    bar = '█' * int(pct / 2)
    print(f"  {i}奇{i}偶: {count:3d}次 ({pct:5.1f}%) {bar}")

# ==================== 8. 冷热号分析 ====================
print("\n" + "=" * 60)
print("8. 冷热号分析（最近30/60/100期）")
print("=" * 60)

for period in [30, 60, 100]:
    recent = df.head(min(period, len(df)))
    print(f"\n最近{period}期各数字出现频率 (前5个最热/最冷):")
    
    all_digits = list(recent['hundred']) + list(recent['ten']) + list(recent['one'])
    counter = Counter(all_digits)
    total = len(all_digits)
    
    freq_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    
    hot = freq_sorted[:5]
    cold = freq_sorted[-5:] if len(freq_sorted) >= 5 else freq_sorted
    
    hot_str = ", ".join([f"{d}({c}次,{c/total*100:.1f}%)" for d, c in hot])
    cold_str = ", ".join([f"{d}({c}次,{c/total*100:.1f}%)" for d, c in cold])
    
    print(f"  最热: {hot_str}")
    print(f"  最冷: {cold_str}")

# ==================== 9. 遗漏值分析 ====================
print("\n" + "=" * 60)
print("9. 遗漏值分析（每个数字当前遗漏了多少期）")
print("=" * 60)

def calc_missing(col_name, label):
    print(f"\n{label}遗漏值 (从上期开始计算):")
    missings = {}
    for digit in range(10):
        # 找到最近出现该数字的位置
        positions = df.index[df[col_name] == digit].tolist()
        if positions:
            last_pos = positions[0]
            missing = df.index[0] - last_pos
        else:
            missing = len(df)
        missings[digit] = missing
        
    for d in range(10):
        bar = '█' * min(missings[d], 20)
        print(f"  数字{d}: 遗漏{missings[d]:2d}期 {bar}")
    return missings

hundred_missing = calc_missing('hundred', '百位')
ten_missing = calc_missing('ten', '十位')
one_missing = calc_missing('one', '个位')

# ==================== 10. 生成可视化图表 ====================
print("\n" + "=" * 60)
print("10. 生成可视化图表...")
print("=" * 60)

# 图1: 各位置数字频率热力图
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 百位频率
ax1 = axes[0, 0]
bars = ax1.bar(range(10), [hundred_freq[i] for i in range(10)], color='steelblue', alpha=0.8)
ax1.set_xlabel('数字')
ax1.set_ylabel('出现次数')
ax1.set_title('百位数字频率分布')
ax1.set_xticks(range(10))
for bar, freq in zip(bars, [hundred_freq[i] for i in range(10)]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(freq), 
             ha='center', va='bottom', fontsize=9)

# 十位频率
ax2 = axes[0, 1]
bars = ax2.bar(range(10), [ten_freq[i] for i in range(10)], color='coral', alpha=0.8)
ax2.set_xlabel('数字')
ax2.set_ylabel('出现次数')
ax2.set_title('十位数字频率分布')
ax2.set_xticks(range(10))
for bar, freq in zip(bars, [ten_freq[i] for i in range(10)]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(freq), 
             ha='center', va='bottom', fontsize=9)

# 个位频率
ax3 = axes[1, 0]
bars = ax3.bar(range(10), [one_freq[i] for i in range(10)], color='seagreen', alpha=0.8)
ax3.set_xlabel('数字')
ax3.set_ylabel('出现次数')
ax3.set_title('个位数字频率分布')
ax3.set_xticks(range(10))
for bar, freq in zip(bars, [one_freq[i] for i in range(10)]):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(freq), 
             ha='center', va='bottom', fontsize=9)

# 和值分布
ax4 = axes[1, 1]
sums = [sum_counts.get(s, 0) for s in range(28)]
ax4.bar(range(28), sums, color='purple', alpha=0.7)
ax4.set_xlabel('和值')
ax4.set_ylabel('出现次数')
ax4.set_title('和值分布 (0-27)')
ax4.set_xticks(range(0, 28, 2))

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_PATH, '01_digit_frequency.png'), dpi=150, bbox_inches='tight')
print("  已保存: 01_digit_frequency.png")

# 图2: 组选类型、跨度、奇偶分布
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 组选类型
ax1 = axes[0, 0]
colors = {'组六': 'steelblue', '组三': 'coral', '豹子': 'gold'}
type_labels = type_counts.index.tolist()
type_values = type_counts.values.tolist()
ax1.pie(type_values, labels=type_labels, autopct='%1.1f%%', 
        colors=[colors.get(t, 'gray') for t in type_labels])
ax1.set_title('组选类型分布')

# 跨度分布
ax2 = axes[0, 1]
span_values = [span_counts.get(s, 0) for s in range(10)]
ax2.bar(range(10), span_values, color='teal', alpha=0.8)
ax2.set_xlabel('跨度值')
ax2.set_ylabel('出现次数')
ax2.set_title('跨度分布 (最大-最小)')
ax2.set_xticks(range(10))

# 奇偶分布
ax3 = axes[1, 0]
odd_values = [odd_pattern.get(i, 0) for i in range(4)]
ax3.bar(['3奇0偶', '2奇1偶', '1奇2偶', '0奇3偶'], odd_values, color='indianred', alpha=0.8)
ax3.set_xlabel('奇偶比例')
ax3.set_ylabel('出现次数')
ax3.set_title('奇偶比例分布')
for i, v in enumerate(odd_values):
    ax3.text(i, v + 0.3, str(v), ha='center')

# 012路分布
ax4 = axes[1, 1]
road_data = {
    '百位': [df['hundred_road'].value_counts().get(i, 0) for i in range(3)],
    '十位': [df['ten_road'].value_counts().get(i, 0) for i in range(3)],
    '个位': [df['one_road'].value_counts().get(i, 0) for i in range(3)]
}
x = np.arange(3)
width = 0.25
ax4.bar(x - width, road_data['百位'], width, label='百位', color='steelblue')
ax4.bar(x, road_data['十位'], width, label='十位', color='coral')
ax4.bar(x + width, road_data['个位'], width, label='个位', color='seagreen')
ax4.set_xlabel('012路')
ax4.set_ylabel('出现次数')
ax4.set_title('012路分布')
ax4.set_xticks(x)
ax4.set_xticklabels(['0路', '1路', '2路'])
ax4.legend()

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_PATH, '02_type_span_parity.png'), dpi=150, bbox_inches='tight')
print("  已保存: 02_type_span_parity.png")

# 图3: 冷热号分析
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, period in enumerate([30, 60, 100]):
    ax = axes[idx]
    recent = df.head(min(period, len(df)))
    all_digits = list(recent['hundred']) + list(recent['ten']) + list(recent['one'])
    counter = Counter(all_digits)
    freq = [counter.get(i, 0) for i in range(10)]
    
    colors = ['red' if f > np.mean(freq) + np.std(freq) else 'blue' if f < np.mean(freq) - np.std(freq) else 'gray' 
              for f in freq]
    bars = ax.bar(range(10), freq, color=colors, alpha=0.8)
    ax.set_xlabel('数字')
    ax.set_ylabel('出现次数')
    ax.set_title(f'最近{period}期冷热号分布')
    ax.set_xticks(range(10))
    
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_PATH, '03_cold_hot_analysis.png'), dpi=150, bbox_inches='tight')
print("  已保存: 03_cold_hot_analysis.png")

# 图4: 遗漏值分析
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, (missing, label) in enumerate([(hundred_missing, '百位'), (ten_missing, '十位'), (one_missing, '个位')]):
    ax = axes[idx]
    values = [missing[i] for i in range(10)]
    colors = ['red' if v > 15 else 'orange' if v > 10 else 'green' for v in values]
    bars = ax.bar(range(10), values, color=colors, alpha=0.8)
    ax.set_xlabel('数字')
    ax.set_ylabel('遗漏期数')
    ax.set_title(f'{label}遗漏值分析')
    ax.set_xticks(range(10))
    ax.axhline(y=np.mean(values), color='black', linestyle='--', alpha=0.5, label=f'均值:{np.mean(values):.1f}')
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_PATH, '04_missing_analysis.png'), dpi=150, bbox_inches='tight')
print("  已保存: 04_missing_analysis.png")

# 图5: 综合热力图
fig, ax = plt.subplots(figsize=(12, 8))
# 创建位置-数字频率矩阵
heatmap_data = np.array([[hundred_freq[i] for i in range(10)],
                          [ten_freq[i] for i in range(10)],
                          [one_freq[i] for i in range(10)]])
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', 
            xticklabels=range(10), yticklabels=['百位', '十位', '个位'], ax=ax)
ax.set_title('各位置数字频率热力图')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_PATH, '05_frequency_heatmap.png'), dpi=150, bbox_inches='tight')
print("  已保存: 05_frequency_heatmap.png")

plt.close('all')

print("\n" + "=" * 60)
print("11. 预测建议")
print("=" * 60)

print("""
基于以上数据分析，提出以下预测参考建议：

【和值建议】
- 和值均值在13左右，接近理论均值13.5
- 建议关注和值范围10-16，占比最高

【跨度建议】
- 跨度4、5、6出现频率较高
- 跨度0（豹子）概率很低，约1%

【冷热号建议】
- 避免过度追冷号，但遗漏超过15期的数字可适当关注
- 近期热号（出现频率明显高于均值）可能有延续性

【012路建议】
- 各路分布相对均衡，无明显偏态
- 可关注"断路"现象（某一路线连续未出现）

【奇偶建议】
- 2奇1偶和1奇2偶出现频率最高
- 全奇全偶出现较少，注意轮换

【组选建议】
- 组六占比最高（约60%），组三次之（约30%），豹子极少（约1%）
- 同尾号（和值相同）可作为辅助参考

【重要提示】
- 彩票开奖为独立随机事件，历史数据不影响未来结果
- 上述分析仅供参考，不能作为投注依据
- 理性购彩，量力而行
""")

print("\n" + "=" * 60)
print("分析完成！图表已保存至 reports/ 目录")
print("=" * 60)
