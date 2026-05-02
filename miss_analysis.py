#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
中国福利彩票3D 遗漏值分析
基于遗漏回归理论：遗漏越久的数字越可能回补
"""

import csv
from collections import defaultdict

# 配置
DATA_FILE = '/Users/wenxin/work/lottory-3d-analysis/data/fc3d_history.csv'
THEORY_AVG_MISS = 10  # 理论平均遗漏值

def load_data():
    """加载历史数据"""
    records = []
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                'issue': row['issue'],
                'hundred': int(row['hundred']),
                'ten': int(row['ten']),
                'one': int(row['one'])
            })
    return records

def calculate_miss(records):
    """
    计算每个位置每个数字的当前遗漏值
    返回: {position: {digit: miss_count}}
    """
    # 初始化：每个位置每个数字的遗漏值
    miss = {
        'hundred': {i: 0 for i in range(10)},
        'ten': {i: 0 for i in range(10)},
        'one': {i: 0 for i in range(10)}
    }
    
    # 从最新一期开始往前计算
    for record in reversed(records):
        h, t, o = record['hundred'], record['ten'], record['one']
        
        # 更新所有数字的遗漏值（未出现的数字遗漏+1）
        for digit in range(10):
            miss['hundred'][digit] += 1
            miss['ten'][digit] += 1
            miss['one'][digit] += 1
        
        # 当前出现的数字重置为0
        miss['hundred'][h] = 0
        miss['ten'][t] = 0
        miss['one'][o] = 0
    
    return miss

def calculate_avg_miss(records):
    """
    计算每个位置每个数字的历史平均遗漏值
    """
    # 记录每个数字每次出现的间隔，计算平均遗漏
    intervals = {
        'hundred': {i: [] for i in range(10)},
        'ten': {i: [] for i in range(10)},
        'one': {i: [] for i in range(10)}
    }
    
    # 记录上次出现的位置
    last_pos = {
        'hundred': {i: None for i in range(10)},
        'ten': {i: None for i in range(10)},
        'one': {i: None for i in range(10)}
    }
    
    for idx, record in enumerate(records):
        h, t, o = record['hundred'], record['ten'], record['one']
        
        for pos, digit in [('hundred', h), ('ten', t), ('one', o)]:
            if last_pos[pos][digit] is not None:
                intervals[pos][digit].append(idx - last_pos[pos][digit])
            last_pos[pos][digit] = idx
    
    # 计算平均遗漏
    avg_miss = {'hundred': {}, 'ten': {}, 'one': {}}
    for pos in ['hundred', 'ten', 'one']:
        for digit in range(10):
            if len(intervals[pos][digit]) > 0:
                avg_miss[pos][digit] = sum(intervals[pos][digit]) / len(intervals[pos][digit])
            else:
                avg_miss[pos][digit] = THEORY_AVG_MISS  # 默认理论值
    
    return avg_miss

def get_top_miss_digits(miss_dict, avg_miss_dict, position, top_n=3):
    """
    获取某个位置遗漏最长的数字
    """
    digits_miss = []
    for digit in range(10):
        current_miss = miss_dict[position][digit]
        avg_miss = avg_miss_dict[position][digit]
        ratio = current_miss / avg_miss if avg_miss > 0 else 0
        digits_miss.append({
            'digit': digit,
            'current_miss': current_miss,
            'avg_miss': avg_miss,
            'ratio': ratio,
            'is_overdue': current_miss > avg_miss * 2  # 超遗漏标记
        })
    
    # 按当前遗漏值排序
    digits_miss.sort(key=lambda x: x['current_miss'], reverse=True)
    return digits_miss[:top_n], digits_miss

def generate_combinations(top_h, top_t, top_o, miss_data):
    """
    生成推荐直选组合
    """
    combinations = []
    
    # 取每个位置前5个数字进行组合
    h_digits = top_h[:5]
    t_digits = top_t[:5]
    o_digits = top_o[:5]
    
    for hd in h_digits:
        for td in t_digits:
            for od in o_digits:
                # 计算组合的综合遗漏分数（越高越推荐）
                score = hd['current_miss'] + td['current_miss'] + od['current_miss']
                overdue_count = (1 if hd['is_overdue'] else 0) + \
                               (1 if td['is_overdue'] else 0) + \
                               (1 if od['is_overdue'] else 0)
                combinations.append({
                    'h': hd['digit'],
                    't': td['digit'],
                    'o': od['digit'],
                    'score': score,
                    'overdue_count': overdue_count,
                    'h_miss': hd['current_miss'],
                    't_miss': td['current_miss'],
                    'o_miss': od['current_miss']
                })
    
    # 排序：优先超遗漏数量，其次综合分数
    combinations.sort(key=lambda x: (x['overdue_count'], x['score']), reverse=True)
    return combinations[:10]

def main():
    print("=" * 60)
    print("【遗漏分析模型结论】")
    print("=" * 60)
    
    # 1. 加载数据
    records = load_data()
    print(f"\n已加载 {len(records)} 期历史数据")
    
    # 2. 计算当前遗漏值
    current_miss = calculate_miss(records)
    
    # 3. 计算历史平均遗漏值
    avg_miss = calculate_avg_miss(records)
    
    # 4. 分析每个位置
    positions = [('hundred', '百位'), ('ten', '十位'), ('one', '个位')]
    top_digits_all = {}
    
    for pos_key, pos_name in positions:
        top_digits, all_digits = get_top_miss_digits(current_miss, avg_miss, pos_key, 10)
        top_digits_all[pos_key] = top_digits
        
        print(f"\n{pos_name}位置 - 当前遗漏排名（理论平均遗漏={THEORY_AVG_MISS}期）:")
        print("-" * 50)
        print(f"{'数字':^6}{'当前遗漏':^10}{'历史平均':^10}{'遗漏倍数':^10}{'状态':^10}")
        print("-" * 50)
        
        for d in top_digits:
            status = "超遗漏!" if d['is_overdue'] else "正常"
            print(f"{d['digit']:^6}{d['current_miss']:^10}{d['avg_miss']:^10.1f}{d['ratio']:^10.1f}{status:^10}")
    
    # 5. 生成推荐组合
    print("\n" + "=" * 60)
    print("【推荐直选组合Top10】")
    print("=" * 60)
    
    combos = generate_combinations(
        top_digits_all['hundred'],
        top_digits_all['ten'],
        top_digits_all['one'],
        current_miss
    )
    
    print(f"\n{'排名':^4}{'组合':^10}{'百位遗漏':^10}{'十位遗漏':^10}{'个位遗漏':^10}{'超遗漏数':^10}{'综合分数':^10}")
    print("-" * 70)
    
    for idx, combo in enumerate(combos, 1):
        combo_str = f"{combo['h']}{combo['t']}{combo['o']}"
        print(f"{idx:^4}{combo_str:^10}{combo['h_miss']:^10}{combo['t_miss']:^10}{combo['o_miss']:^10}{combo['overdue_count']:^10}{combo['score']:^10}")
    
    # 6. 最终预测
    best_combo = combos[0]
    print("\n" + "=" * 60)
    print("【最终预测】")
    print("=" * 60)
    print(f"\n基于遗漏回归理论，推荐直选组合:")
    print(f"\n★ 最终预测：{best_combo['h']}{best_combo['t']}{best_combo['o']} ★")
    print(f"\n预测理由：")
    print(f"  - 百位 {best_combo['h']} 当前遗漏 {best_combo['h_miss']} 期", end="")
    print(" (超遗漏)" if current_miss['hundred'][best_combo['h']] > avg_miss['hundred'][best_combo['h']] * 2 else "")
    print(f"  - 十位 {best_combo['t']} 当前遗漏 {best_combo['t_miss']} 期", end="")
    print(" (超遗漏)" if current_miss['ten'][best_combo['t']] > avg_miss['ten'][best_combo['t']] * 2 else "")
    print(f"  - 个位 {best_combo['o']} 当前遗漏 {best_combo['o_miss']} 期", end="")
    print(" (超遗漏)" if current_miss['one'][best_combo['o']] > avg_miss['one'][best_combo['o']] * 2 else "")
    
    print(f"\n该组合包含 {best_combo['overdue_count']} 个超遗漏数字，综合遗漏分数 {best_combo['score']}")
    print("\n" + "=" * 60)
    print("⚠️  彩票分析仅供参考，请理性投注 ⚠️")
    print("=" * 60)

if __name__ == '__main__':
    main()
