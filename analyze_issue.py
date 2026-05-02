#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""中国福利彩票3D期号规则分析"""

import pandas as pd
import numpy as np
from collections import Counter
import os

# 加载数据
df = pd.read_csv('/Users/wenxin/work/lottory-3d-analysis/data/fc3d_history.csv')
print("=" * 60)
print("数据基本信息")
print("=" * 60)
print(f"总记录数: {len(df)}")
print(f"期号范围: {df['issue'].min()} ~ {df['issue'].max()}")
print(f"日期范围: {df['date'].min()} ~ {df['date'].max()}")
print()

# 期号格式分析
df['issue_str'] = df['issue'].astype(str)
df['prefix_2digit'] = df['issue_str'].str[:2]  # 前2位
df['prefix_3digit'] = df['issue_str'].str[:3]  # 前3位
df['prefix_4digit'] = df['issue_str'].str[:4]  # 前4位（年份）
df['issue_num'] = df['issue_str'].str[4:].astype(int)  # 后3位期号

# ===== 1. 期号前缀规律分析 =====
print("=" * 60)
print("1. 期号前缀规律分析")
print("=" * 60)

print("\n--- 1.1 前2位分布 ---")
prefix2_counts = df['prefix_2digit'].value_counts().sort_index()
for p, c in prefix2_counts.items():
    min_issue = df[df['prefix_2digit']==p]['issue'].min()
    max_issue = df[df['prefix_2digit']==p]['issue'].max()
    print(f"  前缀'{p}': {c}条, 期号范围: {min_issue}-{max_issue}")

print("\n--- 1.2 前3位分布 ---")
prefix3_counts = df['prefix_3digit'].value_counts().sort_index()
for p, c in prefix3_counts.items():
    min_issue = df[df['prefix_3digit']==p]['issue'].min()
    max_issue = df[df['prefix_3digit']==p]['issue'].max()
    print(f"  前缀'{p}': {c}条, 期号范围: {min_issue}-{max_issue}")

print("\n--- 1.3 年份分布(前4位) ---")
year_counts = df['prefix_4digit'].value_counts().sort_index()
for y, c in year_counts.items():
    min_issue = df[df['prefix_4digit']==y]['issue'].min()
    max_issue = df[df['prefix_4digit']==y]['issue'].max()
    print(f"  {y}: {c}条, 期号范围: {min_issue}-{max_issue}")

# 识别不同的期号段
print("\n--- 1.4 期号段划分 ---")
# 基于前3位划分期号段
df['segment'] = df['prefix_3digit']
segments = df['segment'].unique()
print(f"共发现 {len(segments)} 个不同的期号段(前3位): {sorted(segments)}")

for seg in sorted(segments):
    seg_data = df[df['segment'] == seg]
    print(f"\n  期号段 '{seg}':")
    print(f"    记录数: {len(seg_data)}")
    print(f"    期号范围: {seg_data['issue'].min()} - {seg_data['issue'].max()}")
    print(f"    日期范围: {seg_data['date'].min()} - {seg_data['date'].max()}")

# ===== 2. 不同期号段的数字频率对比 =====
print("\n" + "=" * 60)
print("2. 不同期号段的数字频率对比")
print("=" * 60)

def calc_digit_freq(data, digit_name):
    """计算某一位数字的频率分布"""
    counter = Counter(data[digit_name])
    total = len(data)
    freq = {i: counter.get(i, 0) / total * 100 for i in range(10)}
    return freq

segment_stats = {}
for seg in sorted(segments):
    seg_data = df[df['segment'] == seg]
    stats = {
        'count': len(seg_data),
        'hundred': calc_digit_freq(seg_data, 'hundred'),
        'ten': calc_digit_freq(seg_data, 'ten'),
        'one': calc_digit_freq(seg_data, 'one'),
    }
    segment_stats[seg] = stats
    
    print(f"\n--- 期号段 '{seg}' 数字频率分布 ---")
    print(f"  百位: ", end="")
    for d in range(10):
        print(f"{d}:{stats['hundred'][d]:5.1f}% ", end="")
    print()
    print(f"  十位: ", end="")
    for d in range(10):
        print(f"{d}:{stats['ten'][d]:5.1f}% ", end="")
    print()
    print(f"  个位: ", end="")
    for d in range(10):
        print(f"{d}:{stats['one'][d]:5.1f}% ", end="")
    print()

# 找出数字2偏少的期号段
print("\n--- 2.1 数字2偏少的期号段 ---")
for seg in sorted(segments):
    stats = segment_stats[seg]
    h2 = stats['hundred'].get(2, 0)
    t2 = stats['ten'].get(2, 0)
    o2 = stats['one'].get(2, 0)
    avg2 = (h2 + t2 + o2) / 3
    if avg2 < 9.0:  # 理论值是10%
        print(f"  期号段 '{seg}': 百位2出现率={h2:.2f}%, 十位2出现率={t2:.2f}%, 个位2出现率={o2:.2f}%, 平均={avg2:.2f}%")

# ===== 3. 期号与数字的关联分析 =====
print("\n" + "=" * 60)
print("3. 期号与数字的关联分析")
print("=" * 60)

# 期号数值与开奖数字的相关性
print("\n--- 3.1 期号与数字的相关性 ---")
corr_data = []
for seg in sorted(segments):
    seg_data = df[df['segment'] == seg].copy()
    seg_data['issue_num'] = seg_data['issue'] % 1000  # 取期号后三位
    corr_h = seg_data['issue_num'].corr(seg_data['hundred'])
    corr_t = seg_data['issue_num'].corr(seg_data['ten'])
    corr_o = seg_data['issue_num'].corr(seg_data['one'])
    print(f"  期号段 '{seg}': 期号与百位相关={corr_h:.4f}, 与十位相关={corr_t:.4f}, 与个位相关={corr_o:.4f}")

# 高位期号 vs 低位期号的数字分布差异
print("\n--- 3.2 高位期号 vs 低位期号的数字分布差异 ---")
# 以整个数据集的期号中位数为界
median_issue = df['issue'].median()
df['issue_level'] = df['issue'].apply(lambda x: '高位期号' if x >= median_issue else '低位期号')

for level in ['低位期号', '高位期号']:
    level_data = df[df['issue_level'] == level]
    print(f"\n  {level} (期号{'<' if level=='低位' else '>='}{median_issue}): {len(level_data)}条")
    h_freq = calc_digit_freq(level_data, 'hundred')
    t_freq = calc_digit_freq(level_data, 'ten')
    o_freq = calc_digit_freq(level_data, 'one')
    print(f"    百位2出现率: {h_freq.get(2, 0):.2f}%")
    print(f"    十位2出现率: {t_freq.get(2, 0):.2f}%")
    print(f"    个位2出现率: {o_freq.get(2, 0):.2f}%")

# ===== 4. 基于期号段的预测分析 =====
print("\n" + "=" * 60)
print("4. 基于期号段的预测分析")
print("=" * 60)

def simple_predict(train_data):
    """基于频率的简单预测"""
    h_counter = Counter(train_data['hundred'])
    t_counter = Counter(train_data['ten'])
    o_counter = Counter(train_data['one'])
    total = len(train_data)
    pred_h = max(h_counter.keys(), key=lambda x: h_counter[x])
    pred_t = max(t_counter.keys(), key=lambda x: t_counter[x])
    pred_o = max(o_counter.keys(), key=lambda x: o_counter[x])
    return pred_h, pred_t, pred_o

def evaluate_segment(seg_data, name=""):
    """评估预测准确率"""
    if len(seg_data) < 10:
        return None
    # 使用前80%数据训练，后20%测试
    split_idx = int(len(seg_data) * 0.8)
    train = seg_data.iloc[:split_idx]
    test = seg_data.iloc[split_idx:]
    
    pred_h, pred_t, pred_o = simple_predict(train)
    
    # 完全匹配率
    exact_match = ((test['hundred'] == pred_h) & 
                   (test['ten'] == pred_t) & 
                   (test['one'] == pred_o)).sum()
    exact_rate = exact_match / len(test) * 100
    
    # 位置匹配率
    pos_h = (test['hundred'] == pred_h).sum() / len(test) * 100
    pos_t = (test['ten'] == pred_t).sum() / len(test) * 100
    pos_o = (test['one'] == pred_o).sum() / len(test) * 100
    
    return {
        'name': name,
        'count': len(test),
        'exact_rate': exact_rate,
        'pos_h': pos_h,
        'pos_t': pos_t,
        'pos_o': pos_o
    }

print("\n--- 4.1 各期号段预测准确率对比 ---")
predict_results = []
for seg in sorted(segments):
    seg_data = df[df['segment'] == seg].sort_values('date').reset_index(drop=True)
    result = evaluate_segment(seg_data, f"期号段{seg}")
    if result:
        predict_results.append(result)
        print(f"  {result['name']}: 测试样本{result['count']}个, 完全匹配率={result['exact_rate']:.2f}%, "
              f"百位准确率={result['pos_h']:.2f}%, 十位准确率={result['pos_t']:.2f}%, 个位准确率={result['pos_o']:.2f}%")

# 全局对比
all_result = evaluate_segment(df.sort_values('date').reset_index(drop=True), "全部数据")
if all_result:
    print(f"  {all_result['name']}: 测试样本{all_result['count']}个, 完全匹配率={all_result['exact_rate']:.2f}%, "
          f"百位准确率={all_result['pos_h']:.2f}%, 十位准确率={all_result['pos_t']:.2f}%, 个位准确率={all_result['pos_o']:.2f}%")

# ===== 生成分析报告 =====
print("\n" + "=" * 60)
print("生成分析报告")
print("=" * 60)

# 创建输出目录
os.makedirs('/Users/wenxin/work/lottory-3d-analysis/reports/prng_analysis', exist_ok=True)

report = """
================================================================================
                    中国福利彩票3D期号规则分析报告
================================================================================

分析日期: 基于历史数据 {}
数据范围: {} ~ {}
总记录数: {}

================================================================================
一、期号前缀规律分析
================================================================================

1.1 期号结构
   中国福利彩票3D期号为7位数字，格式为: YYY + NNN
   - YYY: 年份后三位 (如2002年记录为"200")
   - NNN: 当年内的期号 (001-999)

1.2 期号前缀分布(前3位)
""".format(
    pd.Timestamp.now().strftime('%Y-%m-%d'),
    df['date'].min(),
    df['date'].max(),
    len(df)
)

for seg in sorted(segments):
    seg_data = df[df['segment'] == seg]
    report += f"""
   期号段 '{seg}':
     - 记录数: {len(seg_data)}
     - 期号范围: {seg_data['issue'].min()} - {seg_data['issue'].max()}
     - 日期范围: {seg_data['date'].min()} - {seg_data['date'].max()}
"""

report += """
================================================================================
二、不同期号段的数字频率对比
================================================================================

"""

for seg in sorted(segments):
    stats = segment_stats[seg]
    report += f"""
2.x 期号段 '{seg}' 数字频率分布
   百位: """
    for d in range(10):
        report += f"{d}:{stats['hundred'][d]:5.1f}% "
    report += "\n   十位: "
    for d in range(10):
        report += f"{d}:{stats['ten'][d]:5.1f}% "
    report += "\n   个位: "
    for d in range(10):
        report += f"{d}:{stats['one'][d]:5.1f}% "
    report += "\n"

report += """
数字2偏少的期号段分析:
"""
for seg in sorted(segments):
    stats = segment_stats[seg]
    h2 = stats['hundred'].get(2, 0)
    t2 = stats['ten'].get(2, 0)
    o2 = stats['one'].get(2, 0)
    avg2 = (h2 + t2 + o2) / 3
    if avg2 < 9.0:
        report += f"  - 期号段 '{seg}': 平均2出现率={avg2:.2f}% (偏低)\n"

report += """
================================================================================
三、期号与数字的关联分析
================================================================================

3.1 期号与数字的相关性分析
"""
for seg in sorted(segments):
    seg_data = df[df['segment'] == seg].copy()
    seg_data['issue_num'] = seg_data['issue'] % 1000
    corr_h = seg_data['issue_num'].corr(seg_data['hundred'])
    corr_t = seg_data['issue_num'].corr(seg_data['ten'])
    corr_o = seg_data['issue_num'].corr(seg_data['one'])
    report += f"   期号段 '{seg}': 与百位相关={corr_h:.4f}, 与十位相关={corr_t:.4f}, 与个位相关={corr_o:.4f}\n"

report += f"""
3.2 高位期号 vs 低位期号数字分布
   分界点: 期号中位数 {median_issue}
"""

for level in ['低位期号', '高位期号']:
    level_data = df[df['issue_level'] == level]
    h_freq = calc_digit_freq(level_data, 'hundred')
    t_freq = calc_digit_freq(level_data, 'ten')
    o_freq = calc_digit_freq(level_data, 'one')
    report += f"""
   {level}: {len(level_data)}条记录
     百位2出现率: {h_freq.get(2, 0):.2f}%
     十位2出现率: {t_freq.get(2, 0):.2f}%
     个位2出现率: {o_freq.get(2, 0):.2f}%
"""

report += """
================================================================================
四、基于期号段的预测分析
================================================================================

使用简单频率统计方法进行预测，将每个期号段内80%数据作为训练集，
20%数据作为测试集，评估预测准确率。

预测准确率对比:
"""
for r in predict_results:
    report += f"   {r['name']}: 完全匹配率={r['exact_rate']:.2f}%, 百位={r['pos_h']:.2f}%, 十位={r['pos_t']:.2f}%, 个位={r['pos_o']:.2f}%\n"

if all_result:
    report += f"   全部数据: 完全匹配率={all_result['exact_rate']:.2f}%, 百位={all_result['pos_h']:.2f}%, 十位={all_result['pos_t']:.2f}%, 个位={all_result['pos_o']:.2f}%\n"

report += """
================================================================================
五、结论与发现
================================================================================

"""

# 找出最佳和最差期号段
if predict_results:
    best = max(predict_results, key=lambda x: x['exact_rate'])
    worst = min(predict_results, key=lambda x: x['exact_rate'])
    report += f"""5.1 预测准确率
   - 最高准确率期号段: {best['name']} (完全匹配率: {best['exact_rate']:.2f}%)
   - 最低准确率期号段: {worst['name']} (完全匹配率: {worst['exact_rate']:.2f}%)
   - 结论: {'不同期号段确实存在预测准确率差异，可能暗示不同期号段使用了不同的随机数生成机制' if best['exact_rate'] != worst['exact_rate'] else '各期号段预测准确率相近'}
"""

# 分析数字2
digit2_low_segments = []
for seg in sorted(segments):
    stats = segment_stats[seg]
    avg2 = (stats['hundred'].get(2, 0) + stats['ten'].get(2, 0) + stats['one'].get(2, 0)) / 3
    if avg2 < 9.0:
        digit2_low_segments.append((seg, avg2))

if digit2_low_segments:
    report += f"""
5.2 数字2偏少现象
   以下期号段存在数字2出现率偏低的现象(平均<9%):
"""
    for seg, rate in digit2_low_segments:
        report += f"   - {seg}: 平均2出现率={rate:.2f}%\n"
    report += "   这可能暗示这些期号段使用了不同的随机数生成算法或设备\n"

# 期号与数字相关性
correlations = []
for seg in sorted(segments):
    seg_data = df[df['segment'] == seg].copy()
    seg_data['issue_num'] = seg_data['issue'] % 1000
    corr_h = abs(seg_data['issue_num'].corr(seg_data['hundred']))
    corr_t = abs(seg_data['issue_num'].corr(seg_data['ten']))
    corr_o = abs(seg_data['issue_num'].corr(seg_data['one']))
    correlations.append((seg, max(corr_h, corr_t, corr_o)))

if correlations:
    max_corr = max(correlations, key=lambda x: x[1])
    report += f"""
5.3 期号与数字关联
   相关性最强的期号段: {max_corr[0]} (最大相关系数: {max_corr[1]:.4f})
   {'存在一定的期号与数字关联，可能存在隐藏模式' if max_corr[1] > 0.1 else '未发现明显的期号与数字关联'}
"""

report += """
================================================================================
                              报告结束
================================================================================
"""

# 保存报告
with open('/Users/wenxin/work/lottory-3d-analysis/reports/prng_analysis/issue_segment_analysis.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("报告已保存到: /Users/wenxin/work/lottory-3d-analysis/reports/prng_analysis/issue_segment_analysis.txt")

# 打印报告内容
print("\n" + report)