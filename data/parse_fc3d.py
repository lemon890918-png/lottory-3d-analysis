#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""解析中国福利彩票3D历史数据"""

import csv
import json
from collections import Counter

# 读取原始数据
with open('/Users/wenxin/work/lottory-3d-analysis/data/3d_asc.txt', 'r') as f:
    lines = f.read().strip().split('\n')

print(f"原始数据行数: {len(lines)}")
print(f"前2行样本:\n{lines[0]}\n{lines[1]}")

# 解析数据
records = []
all_digits = []

for line in lines:
    parts = line.split()
    if len(parts) >= 5:
        issue = parts[0]
        date = parts[1]
        hundred = int(parts[2])
        ten = int(parts[3])
        one = int(parts[4])
        number = hundred * 100 + ten * 10 + one
        records.append({
            'issue': issue,
            'date': date,
            'hundred': hundred,
            'ten': ten,
            'one': one,
            'number': number
        })
        all_digits.extend([hundred, ten, one])

# 统计
total = len(records)
dates = [r['date'] for r in records]
date_range = f"{min(dates)} ~ {max(dates)}"
digit_freq = Counter(all_digits)

# 保存CSV
csv_path = '/Users/wenxin/work/lottory-3d-analysis/data/fc3d_history.csv'
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['issue', 'date', 'hundred', 'ten', 'one', 'number'])
    writer.writeheader()
    writer.writerows(records)
print(f"\nCSV已保存: {csv_path}")

# 保存JSON
json_path = '/Users/wenxin/work/lottory-3d-analysis/data/fc3d_history.json'
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(records, f, ensure_ascii=False, indent=2)
print(f"JSON已保存: {json_path}")

# 输出统计
print("\n" + "="*50)
print("基本统计信息")
print("="*50)
print(f"总期数: {total}")
print(f"日期范围: {date_range}")
print(f"\n0-9各数字出现频率:")
for d in range(10):
    freq = digit_freq.get(d, 0)
    pct = freq / len(all_digits) * 100
    print(f"  {d}: {freq}次 ({pct:.2f}%)")