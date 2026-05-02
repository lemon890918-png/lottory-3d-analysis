#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试组选预测准确率"""

import pandas as pd
import numpy as np
import pickle
from collections import Counter
from sklearn.ensemble import RandomForestClassifier

# 加载数据
df = pd.read_csv('/Users/wenxin/work/lottory-3d-analysis/data/fc3d_history.csv')
df = df.sort_values('issue', ascending=True).reset_index(drop=True)

# 加载模型
with open('/Users/wenxin/work/lottory-3d-analysis/models/rf_hundred.pkl', 'rb') as f:
    rf_hundred = pickle.load(f)
with open('/Users/wenxin/work/lottory-3d-analysis/models/rf_ten.pkl', 'rb') as f:
    rf_ten = pickle.load(f)
with open('/Users/wenxin/work/lottory-3d-analysis/models/rf_one.pkl', 'rb') as f:
    rf_one = pickle.load(f)
with open('/Users/wenxin/work/lottory-3d-analysis/models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 特征工程（必须与训练时完全一致）
N_HISTORY = 10

def create_features(df, n_history=N_HISTORY):
    features = []
    hundred = df['hundred'].values
    ten = df['ten'].values
    one = df['one'].values
    
    for i in range(n_history, len(df)):
        feat = []
        
        # 最近N期的数字编码 (展平)
        for j in range(i - n_history, i):
            feat.extend([hundred[j], ten[j], one[j]])
        
        # 最近N期的和值
        sums = []
        for j in range(i - n_history, i):
            sums.append(hundred[j] + ten[j] + one[j])
        feat.extend(sums)
        
        # 最近N期的跨度
        spans = []
        for j in range(i - n_history, i):
            spans.append(max(hundred[j], ten[j], one[j]) - min(hundred[j], ten[j], one[j]))
        feat.extend(spans)
        
        # 012路特征
        road012 = []
        for j in range(i - n_history, i):
            road012.extend([hundred[j] % 3, ten[j] % 3, one[j] % 3])
        feat.extend(road012)
        
        # 奇偶特征
        parity = []
        for j in range(i - n_history, i):
            parity.extend([hundred[j] % 2, ten[j] % 2, one[j] % 2])
        feat.extend(parity)
        
        # 大小特征
        size = []
        for j in range(i - n_history, i):
            size.extend([1 if hundred[j] >= 5 else 0, 
                        1 if ten[j] >= 5 else 0, 
                        1 if one[j] >= 5 else 0])
        feat.extend(size)
        
        # 当前期的和值、跨度
        feat.append(hundred[i-1] + ten[i-1] + one[i-1])
        feat.append(max(hundred[i-1], ten[i-1], one[i-1]) - min(hundred[i-1], ten[i-1], one[i-1]))
        
        # 历史频率特征
        all_digits = list(hundred[i-n_history:i]) + list(ten[i-n_history:i]) + list(one[i-n_history:i])
        counter = Counter(all_digits)
        for d in range(10):
            feat.append(counter.get(d, 0))
        
        features.append(feat)
    
    return np.array(features)

# 创建特征
X = create_features(df, N_HISTORY)
y_hundred = df['hundred'].values[N_HISTORY:]
y_ten = df['ten'].values[N_HISTORY:]
y_one = df['one'].values[N_HISTORY:]

# 测试集（最后100期）
TEST_SIZE = 100
X_train, X_test = X[:-TEST_SIZE], X[-TEST_SIZE:]
y_h_train, y_h_test = y_hundred[:-TEST_SIZE], y_hundred[-TEST_SIZE:]
y_t_train, y_t_test = y_ten[:-TEST_SIZE], y_ten[-TEST_SIZE:]
y_o_train, y_o_test = y_one[:-TEST_SIZE], y_one[-TEST_SIZE:]

# 训练模型
rf_h = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_t = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_o = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)

rf_h.fit(X_train, y_h_train)
rf_t.fit(X_train, y_t_train)
rf_o.fit(X_train, y_o_train)

# 标准化
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 预测
pred_h = rf_hundred.predict(X_test_scaled)
pred_t = rf_ten.predict(X_test_scaled)
pred_o = rf_one.predict(X_test_scaled)

# 组选判断函数
def get_group_type(h, t, o):
    s = sorted([h, t, o])
    if s[0] == s[2]:
        return '豹子'
    elif s[0] == s[1] or s[1] == s[2]:
        return '组三'
    else:
        return '组六'

def numbers_match(pred_set, real_set):
    """组选：数字集合匹配，顺序不限"""
    return set(pred_set) == set(real_set)

# 统计
direct_hit = 0
group_hit = 0
group_type_hit = 0
total = len(y_h_test)

direct_details = []
group_details = []

for i in range(total):
    real_h = int(y_h_test[i])
    real_t = int(y_t_test[i])
    real_o = int(y_o_test[i])
    real_set = {real_h, real_t, real_o}
    real_type = get_group_type(real_h, real_t, real_o)
    
    pred_h_val = int(pred_h[i])
    pred_t_val = int(pred_t[i])
    pred_o_val = int(pred_o[i])
    pred_set = {pred_h_val, pred_t_val, pred_o_val}
    pred_type = get_group_type(pred_h_val, pred_t_val, pred_o_val)
    
    # 直选
    if pred_h_val == real_h and pred_t_val == real_t and pred_o_val == real_o:
        direct_hit += 1
        direct_details.append(i)
    
    # 组选
    if numbers_match(pred_set, real_set):
        group_hit += 1
        group_details.append((pred_set, real_set, pred_type, real_type))
    
    # 组选+类型
    if set(pred_set) == set(real_set) and pred_type == real_type:
        group_type_hit += 1

print("=" * 60)
print("随机森林模型 预测准确率测试（测试集100期）")
print("=" * 60)
print(f"\n直选（顺序完全匹配）:")
print(f"  命中: {direct_hit}/{total} = {direct_hit}%")

print(f"\n组选（数字集合匹配，顺序不限）:")
print(f"  命中: {group_hit}/{total} = {group_hit}%")

print(f"\n组选+类型（组三/组六/豹子均匹配）:")
print(f"  命中: {group_type_hit}/{total} = {group_type_hit}%")

print(f"\n理论概率参考:")
print(f"  直选: 0.1% | 组选六: 21.6% | 组选三: 2.7% | 豹子: 1.0%")

# 组选类型分布
print(f"\n测试集实际组选类型分布:")
types = [get_group_type(int(y_h_test[i]), int(y_t_test[i]), int(y_o_test[i])) for i in range(total)]
type_counts = Counter(types)
for t, c in type_counts.most_common():
    print(f"  {t}: {c}次 ({c/total*100:.1f}%)")

if group_details:
    print(f"\n组选命中详情（前10条）:")
    for j, (pred, real, pt, rt) in enumerate(group_details[:10]):
        print(f"  #{j+1}: 预测{''.join(map(str,sorted(list(pred))))}({pt}) vs 实际{''.join(map(str,sorted(list(real))))}({rt})")
