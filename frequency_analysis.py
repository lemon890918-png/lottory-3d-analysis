#!/usr/bin/python3
"""
中国福利彩票3D 历史频率分析
基于8619期历史数据，统计百位、十位、个位各自0-9的出现频率
"""

import csv
from collections import Counter

# 读取数据
data_file = '/Users/wenxin/work/lottory-3d-analysis/data/fc3d_history.csv'

hundred_digits = []
ten_digits = []
one_digits = []

with open(data_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        hundred_digits.append(int(row['hundred']))
        ten_digits.append(int(row['ten']))
        one_digits.append(int(row['one']))

print(f"总期数: {len(hundred_digits)}")

# 统计各位置0-9出现频率
hundred_counter = Counter(hundred_digits)
ten_counter = Counter(ten_digits)
one_counter = Counter(one_digits)

print("\n===== 百位频率统计 =====")
total = len(hundred_digits)
for digit in range(10):
    count = hundred_counter.get(digit, 0)
    freq = count / total * 100
    print(f"数字{digit}: 出现{count}次, 频率{freq:.2f}%")

print("\n===== 十位频率统计 =====")
for digit in range(10):
    count = ten_counter.get(digit, 0)
    freq = count / total * 100
    print(f"数字{digit}: 出现{count}次, 频率{freq:.2f}%")

print("\n===== 个位频率统计 =====")
for digit in range(10):
    count = one_counter.get(digit, 0)
    freq = count / total * 100
    print(f"数字{digit}: 出现{count}次, 频率{freq:.2f}%")

# 找出每个位置出现频率最高的3个数字
hundred_top3 = hundred_counter.most_common(3)
ten_top3 = ten_counter.most_common(3)
one_top3 = one_counter.most_common(3)

print("\n===== 各位置Top3高频数字 =====")
print(f"百位Top3: {[(d, c, c/total*100) for d, c in hundred_top3]}")
print(f"十位Top3: {[(d, c, c/total*100) for d, c in ten_top3]}")
print(f"个位Top3: {[(d, c, c/total*100) for d, c in one_top3]}")

# 生成推荐直选组合Top10
# 策略：按各位置频率权重生成组合
# 先计算每个位置的概率权重
hundred_probs = {d: hundred_counter.get(d, 0) / total for d in range(10)}
ten_probs = {d: ten_counter.get(d, 0) / total for d in range(10)}
one_probs = {d: one_counter.get(d, 0) / total for d in range(10)}

# 生成所有可能的组合并计算联合概率
# 由于组合太多(1000种)，我们使用Top3组合的笛卡尔积
combinations = []
for h, h_cnt in hundred_top3:
    for t, t_cnt in ten_top3:
        for o, o_cnt in one_top3:
            # 理论概率 = 各位置概率乘积
            prob = (h_cnt / total) * (t_cnt / total) * (o_cnt / total)
            combinations.append((h, t, o, prob))

# 按概率排序
combinations.sort(key=lambda x: x[3], reverse=True)

print("\n===== 推荐直选组合Top10 =====")
for i, (h, t, o, prob) in enumerate(combinations[:10], 1):
    print(f"Top{i}: {h}{t}{o}, 理论概率={prob:.6f} ({prob*100:.4f}%)")

# 最终预测：选择概率最高的组合
best = combinations[0]
print(f"\n===== 最终预测 =====")
print(f"基于历史频率分析，推荐直选: {best[0]}{best[1]}{best[2]}")
print(f"理论概率: {best[3]*100:.4f}%")

# 输出百位、十位、个位的推荐
print("\n===== 各位置推荐 =====")
print(f"百位推荐: {[d for d, c in hundred_top3]} (理由: 历史出现频率最高)")
print(f"十位推荐: {[d for d, c in ten_top3]} (理由: 历史出现频率最高)")
print(f"个位推荐: {[d for d, c in one_top3]} (理由: 历史出现频率最高)")