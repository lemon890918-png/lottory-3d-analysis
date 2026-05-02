#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中国福利彩票3D预测模型
包含三种模型：统计模型、机器学习模型、深度学习模型(LSTM)
"""

import pandas as pd
import numpy as np
import json
import os
import pickle
from datetime import datetime
from collections import Counter, defaultdict

# 机器学习
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 深度学习
import warnings
warnings.filterwarnings('ignore')

# 尝试导入TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
    print(f"TensorFlow版本: {tf.__version__}")
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow不可用，深度学习模型将被跳过")

# 设置随机种子
np.random.seed(42)
if TF_AVAILABLE:
    tf.random.set_seed(42)

# 路径配置
PROJECT_PATH = "/Users/wenxin/work/lottory-3d-analysis"
DATA_PATH = f"{PROJECT_PATH}/data/fc3d_history.csv"
MODEL_PATH = f"{PROJECT_PATH}/models"
REPORT_PATH = f"{PROJECT_PATH}/reports"

# 创建目录
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(REPORT_PATH, exist_ok=True)

print("=" * 60)
print("中国福利彩票3D预测系统")
print("=" * 60)

# =============================================================================
# 1. 数据加载与预处理
# =============================================================================
print("\n[1] 加载历史数据...")

df = pd.read_csv(DATA_PATH)
df = df.sort_values('issue', ascending=True).reset_index(drop=True)
print(f"总共加载 {len(df)} 期数据")
print(f"数据范围: {df['issue'].min()} - {df['issue'].max()}")
print(f"日期范围: {df['date'].min()} - {df['date'].max()}")

# 提取百位、十位、个位
hundred = df['hundred'].values
ten = df['ten'].values  
one = df['one'].values

print(f"\n百位分布: {Counter(hundred).most_common(5)}")
print(f"十位分布: {Counter(ten).most_common(5)}")
print(f"个位分布: {Counter(one).most_common(5)}")

# =============================================================================
# 2. 特征工程
# =============================================================================
print("\n[2] 构建特征...")

def create_features(df, n_history=10):
    """创建用于机器学习模型的特征"""
    features = []
    labels_hundred = []
    labels_ten = []
    labels_one = []
    
    hundred = df['hundred'].values
    ten = df['ten'].values
    one = df['one'].values
    number = df['number'].values
    
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
        
        # 最近N期的跨度 (max - min)
        spans = []
        for j in range(i - n_history, i):
            spans.append(max(hundred[j], ten[j], one[j]) - min(hundred[j], ten[j], one[j]))
        feat.extend(spans)
        
        # 012路特征 (每个数字对3取模)
        road012 = []
        for j in range(i - n_history, i):
            road012.extend([hundred[j] % 3, ten[j] % 3, one[j] % 3])
        feat.extend(road012)
        
        # 奇偶特征
        parity = []
        for j in range(i - n_history, i):
            parity.extend([hundred[j] % 2, ten[j] % 2, one[j] % 2])
        feat.extend(parity)
        
        # 大小特征 (0-4小, 5-9大)
        size = []
        for j in range(i - n_history, i):
            size.extend([1 if hundred[j] >= 5 else 0, 
                        1 if ten[j] >= 5 else 0, 
                        1 if one[j] >= 5 else 0])
        feat.extend(size)
        
        # 当前期的和值、跨度
        feat.append(hundred[i-1] + ten[i-1] + one[i-1])
        feat.append(max(hundred[i-1], ten[i-1], one[i-1]) - min(hundred[i-1], ten[i-1], one[i-1]))
        
        # 历史频率特征 (最近N期)
        all_digits = list(hundred[i-n_history:i]) + list(ten[i-n_history:i]) + list(one[i-n_history:i])
        counter = Counter(all_digits)
        for d in range(10):
            feat.append(counter.get(d, 0))
        
        features.append(feat)
        labels_hundred.append(hundred[i])
        labels_ten.append(ten[i])
        labels_one.append(one[i])
    
    return np.array(features), np.array(labels_hundred), np.array(labels_ten), np.array(labels_one)

# 创建特征
N_HISTORY = 10
X, y_hundred, y_ten, y_one = create_features(df, N_HISTORY)
print(f"特征维度: {X.shape}")

# 划分训练集和测试集 (最后100期为测试集)
TEST_SIZE = 100
X_train, X_test = X[:-TEST_SIZE], X[-TEST_SIZE:]
y_h_train, y_h_test = y_hundred[:-TEST_SIZE], y_hundred[-TEST_SIZE:]
y_t_train, y_t_test = y_ten[:-TEST_SIZE], y_ten[-TEST_SIZE:]
y_o_train, y_o_test = y_one[:-TEST_SIZE], y_one[-TEST_SIZE:]

print(f"训练集: {len(X_train)} 样本")
print(f"测试集: {len(X_test)} 样本")

# =============================================================================
# 3. 统计模型
# =============================================================================
print("\n[3] 训练统计模型...")

class StatisticalModel:
    """基于频率、遗漏值、冷热号的统计预测模型"""
    
    def __init__(self):
        self.name = "统计模型"
        
    def fit(self, hundred, ten, one):
        """训练模型 - 实际上只是计算统计量"""
        self.hundred_freq = Counter(hundred)
        self.ten_freq = Counter(ten)
        self.one_freq = Counter(one)
        
        # 计算遗漏值
        self.hundred_miss = {i: 0 for i in range(10)}
        self.ten_miss = {i: 0 for i in range(10)}
        self.one_miss = {i: 0 for i in range(10)}
        
        for h in hundred:
            for d in range(10):
                self.hundred_miss[d] += 1
            self.hundred_miss[h] = 0
            
        for t in ten:
            for d in range(10):
                self.ten_miss[d] += 1
            self.ten_miss[t] = 0
            
        for o in one:
            for d in range(10):
                self.one_miss[d] += 1
            self.one_miss[o] = 0
    
    def predict(self):
        """预测下一期 - 基于频率和遗漏值综合评分"""
        scores_h = {}
        scores_t = {}
        scores_o = {}
        
        total = sum(self.hundred_freq.values())
        
        for d in range(10):
            # 频率得分 (出现次数越多得分越高)
            freq_h = self.hundred_freq.get(d, 0) / total
            freq_t = self.ten_freq.get(d, 0) / total
            freq_o = self.one_freq.get(d, 0) / total
            
            # 遗漏值得分 (遗漏越大得分越高，期望回补)
            miss_h = self.hundred_miss[d] / 100
            miss_t = self.ten_miss[d] / 100
            miss_o = self.one_miss[d] / 100
            
            # 综合得分 (频率和遗漏各占50%)
            scores_h[d] = 0.5 * freq_h + 0.5 * miss_h
            scores_t[d] = 0.5 * freq_t + 0.5 * miss_t
            scores_o[d] = 0.5 * freq_o + 0.5 * miss_o
        
        # 归一化为概率
        sum_h = sum(scores_h.values())
        sum_t = sum(scores_t.values())
        sum_o = sum(scores_o.values())
        
        prob_h = {d: scores_h[d]/sum_h for d in range(10)}
        prob_t = {d: scores_t[d]/sum_t for d in range(10)}
        prob_o = {d: scores_o[d]/sum_o for d in range(10)}
        
        # 取最高概率的作为预测
        pred_h = max(prob_h.keys(), key=lambda x: prob_h[x])
        pred_t = max(prob_t.keys(), key=lambda x: prob_t[x])
        pred_o = max(prob_o.keys(), key=lambda x: prob_o[x])
        
        return (pred_h, pred_t, pred_o), (prob_h, prob_t, prob_o)
    
    def evaluate(self, hundred, ten, one):
        """评估模型准确率"""
        correct = 0
        for i in range(len(hundred)):
            pred, _ = self.predict()
            if pred == (hundred[i], ten[i], one[i]):
                correct += 1
        return correct / len(hundred)

stat_model = StatisticalModel()
stat_model.fit(hundred, ten, one)

# 在测试集上评估
stat_correct = 0
stat_results = []
for i in range(len(df) - TEST_SIZE, len(df)):
    pred, prob = stat_model.predict()
    actual = (df.iloc[i]['hundred'], df.iloc[i]['ten'], df.iloc[i]['one'])
    stat_results.append({
        'issue': df.iloc[i]['issue'],
        'pred_h': pred[0], 'pred_t': pred[1], 'pred_o': pred[2],
        'actual_h': actual[0], 'actual_t': actual[1], 'actual_o': actual[2],
        'hit': pred == actual
    })
    if pred == actual:
        stat_correct += 1
    # 更新遗漏值
    stat_model.hundred_miss[actual[0]] += 1
    stat_model.ten_miss[actual[1]] += 1
    stat_model.one_miss[actual[2]] += 1
    stat_model.hundred_miss[pred[0]] = 0
    stat_model.ten_miss[pred[1]] = 0
    stat_model.one_miss[pred[2]] = 0

stat_accuracy = stat_correct / TEST_SIZE
print(f"统计模型测试集准确率: {stat_accuracy:.4f} ({stat_correct}/{TEST_SIZE})")

# 统计模型预测下一期
stat_pred, stat_prob = stat_model.predict()
print(f"\n统计模型预测下一期: 百位={stat_pred[0]}, 十位={stat_pred[1]}, 个位={stat_pred[2]}")

# =============================================================================
# 4. 机器学习模型 (随机森林 + 梯度提升)
# =============================================================================
print("\n[4] 训练机器学习模型...")

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练随机森林模型
print("训练随机森林模型...")
rf_h = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_t = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_o = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)

rf_h.fit(X_train_scaled, y_h_train)
rf_t.fit(X_train_scaled, y_t_train)
rf_o.fit(X_train_scaled, y_o_train)

# 预测概率
rf_pred_h = rf_h.predict(X_test_scaled)
rf_pred_t = rf_t.predict(X_test_scaled)
rf_pred_o = rf_o.predict(X_test_scaled)

rf_prob_h = rf_h.predict_proba(X_test_scaled)
rf_prob_t = rf_t.predict_proba(X_test_scaled)
rf_prob_o = rf_o.predict_proba(X_test_scaled)

# 准确率
rf_acc_h = accuracy_score(y_h_test, rf_pred_h)
rf_acc_t = accuracy_score(y_t_test, rf_pred_t)
rf_acc_o = accuracy_score(y_o_test, rf_pred_o)
rf_acc_exact = sum(rf_pred_h == y_h_test) & (rf_pred_t == y_t_test) & (rf_pred_o == y_o_test)

print(f"随机森林 - 百位准确率: {rf_acc_h:.4f}")
print(f"随机森林 - 十位准确率: {rf_acc_t:.4f}")
print(f"随机森林 - 个位准确率: {rf_acc_o:.4f}")

# 完全匹配准确率
rf_exact_match = 0
for i in range(len(y_h_test)):
    if rf_pred_h[i] == y_h_test[i] and rf_pred_t[i] == y_t_test[i] and rf_pred_o[i] == y_o_test[i]:
        rf_exact_match += 1
rf_exact_accuracy = rf_exact_match / TEST_SIZE
print(f"随机森林 - 完全匹配准确率: {rf_exact_accuracy:.4f} ({rf_exact_match}/{TEST_SIZE})")

# 机器学习模型预测下一期
last_features = X[-1:].copy()
last_features_scaled = scaler.transform(last_features)

ml_pred_h = rf_h.predict(last_features_scaled)[0]
ml_pred_t = rf_t.predict(last_features_scaled)[0]
ml_pred_o = rf_o.predict(last_features_scaled)[0]

ml_prob_h = rf_prob_h[0]
ml_prob_t = rf_prob_t[0]
ml_prob_o = rf_prob_o[0]

print(f"\n机器学习模型预测下一期: 百位={ml_pred_h}, 十位={ml_pred_t}, 个位={ml_pred_o}")

# 保存模型
with open(f"{MODEL_PATH}/scaler.pkl", 'wb') as f:
    pickle.dump(scaler, f)
with open(f"{MODEL_PATH}/rf_hundred.pkl", 'wb') as f:
    pickle.dump(rf_h, f)
with open(f"{MODEL_PATH}/rf_ten.pkl", 'wb') as f:
    pickle.dump(rf_t, f)
with open(f"{MODEL_PATH}/rf_one.pkl", 'wb') as f:
    pickle.dump(rf_o, f)

# =============================================================================
# 5. 深度学习模型 (LSTM)
# =============================================================================
if TF_AVAILABLE:
    print("\n[5] 训练深度学习模型(LSTM)...")
    
    # 为LSTM准备序列数据
    def create_sequences(hundred, ten, one, seq_length=20):
        X_seq = []
        y_seq_h = []
        y_seq_t = []
        y_seq_o = []
        
        for i in range(seq_length, len(hundred)):
            # 形状: (seq_length, 3) - 3个数字
            seq = np.column_stack([
                hundred[i-seq_length:i],
                ten[i-seq_length:i],
                one[i-seq_length:i]
            ])
            X_seq.append(seq)
            y_seq_h.append(hundred[i])
            y_seq_t.append(ten[i])
            y_seq_o.append(one[i])
        
        return np.array(X_seq), np.array(y_seq_h), np.array(y_seq_t), np.array(y_seq_o)
    
    SEQ_LENGTH = 20
    X_seq, y_seq_h, y_seq_t, y_seq_o = create_sequences(hundred, ten, one, SEQ_LENGTH)
    
    print(f"LSTM序列数据形状: {X_seq.shape}")
    
    # 划分数据集
    X_seq_train = X_seq[:-TEST_SIZE]
    X_seq_test = X_seq[-TEST_SIZE:]
    y_seq_h_train = y_seq_h[:-TEST_SIZE]
    y_seq_h_test = y_seq_h[-TEST_SIZE:]
    y_seq_t_train = y_seq_t[:-TEST_SIZE]
    y_seq_t_test = y_seq_t[-TEST_SIZE:]
    y_seq_o_train = y_seq_o[:-TEST_SIZE]
    y_seq_o_test = y_seq_o[-TEST_SIZE:]
    
    # One-hot编码
    y_seq_h_train_oh = keras.utils.to_categorical(y_seq_h_train, 10)
    y_seq_h_test_oh = keras.utils.to_categorical(y_seq_h_test, 10)
    y_seq_t_train_oh = keras.utils.to_categorical(y_seq_t_train, 10)
    y_seq_t_test_oh = keras.utils.to_categorical(y_seq_t_test, 10)
    y_seq_o_train_oh = keras.utils.to_categorical(y_seq_o_train, 10)
    y_seq_o_test_oh = keras.utils.to_categorical(y_seq_o_test, 10)
    
    # 构建LSTM模型
    def build_lstm_model(seq_length):
        model = Sequential([
            Input(shape=(seq_length, 3)),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    # 训练早停
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    print("训练LSTM百位模型...")
    lstm_h = build_lstm_model(SEQ_LENGTH)
    lstm_h.fit(X_seq_train, y_seq_h_train_oh, 
               epochs=50, batch_size=32, validation_split=0.1,
               callbacks=[early_stop], verbose=0)
    
    print("训练LSTM十位模型...")
    lstm_t = build_lstm_model(SEQ_LENGTH)
    lstm_t.fit(X_seq_train, y_seq_t_train_oh,
               epochs=50, batch_size=32, validation_split=0.1,
               callbacks=[early_stop], verbose=0)
    
    print("训练LSTM个位模型...")
    lstm_o = build_lstm_model(SEQ_LENGTH)
    lstm_o.fit(X_seq_train, y_seq_o_train_oh,
               epochs=50, batch_size=32, validation_split=0.1,
               callbacks=[early_stop], verbose=0)
    
    # 评估LSTM模型
    lstm_pred_h = np.argmax(lstm_h.predict(X_seq_test, verbose=0), axis=1)
    lstm_pred_t = np.argmax(lstm_t.predict(X_seq_test, verbose=0), axis=1)
    lstm_pred_o = np.argmax(lstm_o.predict(X_seq_test, verbose=0), axis=1)
    
    lstm_acc_h = accuracy_score(y_seq_h_test, lstm_pred_h)
    lstm_acc_t = accuracy_score(y_seq_t_test, lstm_pred_t)
    lstm_acc_o = accuracy_score(y_seq_o_test, lstm_pred_o)
    
    print(f"LSTM - 百位准确率: {lstm_acc_h:.4f}")
    print(f"LSTM - 十位准确率: {lstm_acc_t:.4f}")
    print(f"LSTM - 个位准确率: {lstm_acc_o:.4f}")
    
    lstm_exact_match = 0
    for i in range(len(y_seq_h_test)):
        if lstm_pred_h[i] == y_seq_h_test[i] and lstm_pred_t[i] == y_seq_t_test[i] and lstm_pred_o[i] == y_seq_o_test[i]:
            lstm_exact_match += 1
    lstm_exact_accuracy = lstm_exact_match / len(y_seq_h_test)
    print(f"LSTM - 完全匹配准确率: {lstm_exact_accuracy:.4f} ({lstm_exact_match}/{len(y_seq_h_test)})")
    
    # LSTM预测下一期
    last_seq = X_seq[-1:].copy()
    lstm_pred_next_h = np.argmax(lstm_h.predict(last_seq, verbose=0), axis=1)[0]
    lstm_pred_next_t = np.argmax(lstm_t.predict(last_seq, verbose=0), axis=1)[0]
    lstm_pred_next_o = np.argmax(lstm_o.predict(last_seq, verbose=0), axis=1)[0]
    
    lstm_prob_h = lstm_h.predict(last_seq, verbose=0)[0]
    lstm_prob_t = lstm_t.predict(last_seq, verbose=0)[0]
    lstm_prob_o = lstm_o.predict(last_seq, verbose=0)[0]
    
    print(f"\nLSTM模型预测下一期: 百位={lstm_pred_next_h}, 十位={lstm_pred_next_t}, 个位={lstm_pred_next_o}")
    
    # 保存LSTM模型
    lstm_h.save(f"{MODEL_PATH}/lstm_hundred.h5")
    lstm_t.save(f"{MODEL_PATH}/lstm_ten.h5")
    lstm_o.save(f"{MODEL_PATH}/lstm_one.h5")
    print("LSTM模型已保存")
else:
    lstm_exact_accuracy = 0
    lstm_pred_next_h, lstm_pred_next_t, lstm_pred_next_o = 0, 0, 0

# =============================================================================
# 6. 生成预测报告
# =============================================================================
print("\n[6] 生成预测报告...")

report = []
report.append("=" * 70)
report.append("中国福利彩票3D预测分析报告")
report.append("=" * 70)
report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report.append(f"数据来源: {DATA_PATH}")
report.append(f"数据总量: {len(df)} 期")
report.append(f"训练集: {len(df) - TEST_SIZE} 期")
report.append(f"测试集: {TEST_SIZE} 期")
report.append("")

report.append("-" * 70)
report.append("一、各模型预测结果")
report.append("-" * 70)

report.append("")
report.append("1. 统计模型 (基于频率、遗漏值、冷热号)")
report.append(f"   预测结果: 百位={stat_pred[0]}, 十位={stat_pred[1]}, 个位={stat_pred[2]}")
report.append(f"   测试集准确率: {stat_accuracy:.4f} ({stat_correct}/{TEST_SIZE})")
report.append("   预测概率分布:")
report.append(f"   百位: {dict(sorted(stat_prob[0].items(), key=lambda x: x[1], reverse=True)[:5])}")
report.append(f"   十位: {dict(sorted(stat_prob[1].items(), key=lambda x: x[1], reverse=True)[:5])}")
report.append(f"   个位: {dict(sorted(stat_prob[2].items(), key=lambda x: x[1], reverse=True)[:5])}")

report.append("")
report.append("2. 机器学习模型 (随机森林)")
report.append(f"   预测结果: 百位={ml_pred_h}, 十位={ml_pred_t}, 个位={ml_pred_o}")
report.append(f"   测试集准确率:")
report.append(f"   - 百位: {rf_acc_h:.4f}")
report.append(f"   - 十位: {rf_acc_t:.4f}")
report.append(f"   - 个位: {rf_acc_o:.4f}")
report.append(f"   - 完全匹配: {rf_exact_accuracy:.4f} ({rf_exact_match}/{TEST_SIZE})")

if TF_AVAILABLE:
    report.append("")
    report.append("3. 深度学习模型 (LSTM)")
    report.append(f"   预测结果: 百位={lstm_pred_next_h}, 十位={lstm_pred_next_t}, 个位={lstm_pred_next_o}")
    report.append(f"   测试集准确率:")
    report.append(f"   - 百位: {lstm_acc_h:.4f}")
    report.append(f"   - 十位: {lstm_acc_t:.4f}")
    report.append(f"   - 个位: {lstm_acc_o:.4f}")
    report.append(f"   - 完全匹配: {lstm_exact_accuracy:.4f} ({lstm_exact_match}/{len(y_seq_h_test)})")

report.append("")
report.append("-" * 70)
report.append("二、模型性能对比")
report.append("-" * 70)
report.append("")
report.append(f"{'模型':<20} {'百位准确率':<15} {'十位准确率':<15} {'个位准确率':<15} {'完全匹配率':<15}")
report.append("-" * 70)
report.append(f"{'统计模型':<20} {'N/A':<15} {'N/A':<15} {'N/A':<15} {stat_accuracy:.4f}")
report.append(f"{'随机森林':<20} {rf_acc_h:<15.4f} {rf_acc_t:<15.4f} {rf_acc_o:<15.4f} {rf_exact_accuracy:<15.4f}")
if TF_AVAILABLE:
    report.append(f"{'LSTM':<20} {lstm_acc_h:<15.4f} {lstm_acc_t:<15.4f} {lstm_acc_o:<15.4f} {lstm_exact_accuracy:<15.4f}")

report.append("")
report.append("-" * 70)
report.append("三、综合预测建议")
report.append("-" * 70)
report.append("")
report.append("基于三个模型的预测结果，综合分析如下:")
report.append("")

# 综合预测逻辑
all_preds = [
    (stat_pred, "统计模型"),
    ((ml_pred_h, ml_pred_t, ml_pred_o), "随机森林")
]
if TF_AVAILABLE:
    all_preds.append(((lstm_pred_next_h, lstm_pred_next_t, lstm_pred_next_o), "LSTM"))

# 投票
votes_h = Counter([p[0] for p, _ in all_preds])
votes_t = Counter([p[1] for p, _ in all_preds])
votes_o = Counter([p[2] for p, _ in all_preds])

final_h = votes_h.most_common(1)[0][0]
final_t = votes_t.most_common(1)[0][0]
final_o = votes_o.most_common(1)[0][0]

report.append(f"投票结果: 百位={dict(votes_h)}, 十位={dict(votes_t)}, 个位={dict(votes_o)}")
report.append(f"综合推荐: 百位={final_h}, 十位={final_t}, 个位={final_o}")
report.append("")

# 频率分析
report.append("近期热门数字分析:")
recent = df.tail(20)
report.append(f"百位热门: {Counter(recent['hundred']).most_common(3)}")
report.append(f"十位热门: {Counter(recent['ten']).most_common(3)}")
report.append(f"个位热门: {Counter(recent['one']).most_common(3)}")

report.append("")
report.append("-" * 70)
report.append("四、风险提示")
report.append("-" * 70)
report.append("")
report.append("重要声明:")
report.append("1. 彩票开奖是完全随机的独立事件，任何预测模型都无法保证准确性")
report.append("2. 历史数据中的「规律」仅是统计意义上的相关性，不代表因果关系")
report.append("3. 本模型的预测结果仅供参考，请理性购彩，量力而行")
report.append("4. 期望值约为0.001 (直选)，请勿沉迷")
report.append("")
report.append(f"直选理论概率: 1/1000 = 0.001 (0.1%)")
report.append(f"组三理论概率: 3/1000 = 0.003 (0.3%)")
report.append(f"组六理论概率: 6/1000 = 0.006 (0.6%)")

report.append("")
report.append("-" * 70)
report.append("五、模型文件")
report.append("-" * 70)
report.append(f"模型保存路径: {MODEL_PATH}/")
report.append("- scaler.pkl: 特征标准化器")
report.append("- rf_hundred.pkl: 随机森林百位模型")
report.append("- rf_ten.pkl: 随机森林十位模型")
report.append("- rf_one.pkl: 随机森林个位模型")
if TF_AVAILABLE:
    report.append("- lstm_hundred.h5: LSTM百位模型")
    report.append("- lstm_ten.h5: LSTM十位模型")
    report.append("- lstm_one.h5: LSTM个位模型")

report.append("")
report.append("=" * 70)
report.append("报告结束")
report.append("=" * 70)

# 保存报告
report_text = "\n".join(report)
report_file = f"{REPORT_PATH}/prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
with open(report_file, 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f"报告已保存到: {report_file}")

# 保存最新预测结果到JSON
latest_prediction = {
    "timestamp": datetime.now().isoformat(),
    "statistical_model": {
        "prediction": [int(stat_pred[0]), int(stat_pred[1]), int(stat_pred[2])],
        "probabilities": {
            "hundred": {str(k): round(float(v), 4) for k, v in stat_prob[0].items()},
            "ten": {str(k): round(float(v), 4) for k, v in stat_prob[1].items()},
            "one": {str(k): round(float(v), 4) for k, v in stat_prob[2].items()}
        },
        "test_accuracy": round(float(stat_accuracy), 4)
    },
    "random_forest": {
        "prediction": [int(ml_pred_h), int(ml_pred_t), int(ml_pred_o)],
        "test_accuracy": {
            "hundred": round(float(rf_acc_h), 4),
            "ten": round(float(rf_acc_t), 4),
            "one": round(float(rf_acc_o), 4),
            "exact_match": round(float(rf_exact_accuracy), 4)
        }
    },
    "lstm": {
        "available": TF_AVAILABLE,
        "prediction": [int(lstm_pred_next_h), int(lstm_pred_next_t), int(lstm_pred_next_o)] if TF_AVAILABLE else None,
        "test_accuracy": {
            "hundred": round(float(lstm_acc_h), 4),
            "ten": round(float(lstm_acc_t), 4),
            "one": round(float(lstm_acc_o), 4),
            "exact_match": round(float(lstm_exact_accuracy), 4)
        } if TF_AVAILABLE else None
    },
    "ensemble": {
        "prediction": [int(final_h), int(final_t), int(final_o)],
        "voting": {
            "hundred": {str(k): int(v) for k, v in votes_h.items()},
            "ten": {str(k): int(v) for k, v in votes_t.items()},
            "one": {str(k): int(v) for k, v in votes_o.items()}
        }
    }
}

json_file = f"{REPORT_PATH}/latest_prediction.json"
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(latest_prediction, f, ensure_ascii=False, indent=2)

print(f"预测结果JSON已保存到: {json_file}")

print("\n" + "=" * 60)
print("预测完成!")
print("=" * 60)
print(report_text)