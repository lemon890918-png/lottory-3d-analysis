#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中国福利彩票3D PRNG统计检验
对8619期历史数据进行随机性分析
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2, kstest
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import os
import warnings
warnings.filterwarnings('ignore')

# 设置路径
DATA_PATH = '/Users/wenxin/work/lottory-3d-analysis/data/fc3d_history.csv'
OUTPUT_DIR = '/Users/wenxin/work/lottory-3d-analysis/reports/prng_analysis'

def load_data():
    """加载数据"""
    print("=" * 60)
    print("1. 数据加载")
    print("=" * 60)
    df = pd.read_csv(DATA_PATH)
    print(f"总期数: {len(df)}")
    print(f"数据列: {list(df.columns)}")
    print(f"时间范围: {df['date'].iloc[0]} 至 {df['date'].iloc[-1]}")
    return df

def chi_square_test(data, name):
    """卡方检验 - 检验0-9均匀分布"""
    print(f"\n{'=' * 60}")
    print(f"2. 卡方检验 (Chi-Square) - {name}")
    print("=" * 60)
    
    observed = pd.Series(data).value_counts().sort_index()
    expected = len(data) / 10
    
    # 确保所有0-9都有计数
    for i in range(10):
        if i not in observed.index:
            observed[i] = 0
    observed = observed.sort_index()
    
    # 计算卡方统计量
    chi2_stat = sum((observed.values - expected) ** 2 / expected)
    df = 9  # 自由度为9 (10个类别-1)
    p_value = 1 - chi2.cdf(chi2_stat, df)
    
    print(f"观察频数: {observed.values}")
    print(f"期望频数: {expected:.2f}")
    print(f"卡方统计量: {chi2_stat:.4f}")
    print(f"自由度: {df}")
    print(f"p值: {p_value:.6f}")
    
    # 检验每个数字的偏差
    print("\n各数字偏差分析:")
    abnormal = []
    for i in range(10):
        obs = observed[i]
        deviation = (obs - expected) / expected * 100
        status = ""
        if abs(deviation) > 5:
            status = "【异常】"
            abnormal.append(i)
        elif abs(deviation) > 3:
            status = "【可疑】"
        print(f"  数字{i}: 观察={obs}, 偏差={deviation:+.2f}% {status}")
    
    conclusion = "通过" if p_value > 0.05 else "异常"
    if abnormal:
        conclusion = "异常"
    print(f"\n结论: {conclusion} (α=0.05)")
    
    return chi2_stat, p_value, conclusion, abnormal

def runs_test(data, name):
    """游程检验"""
    print(f"\n{'=' * 60}")
    print(f"3. 游程检验 (Runs Test) - {name}")
    print("=" * 60)
    
    data = np.array(data)
    median = np.median(data)
    
    # 创建二元序列 (高于中位数=1, 低于或等于=0)
    binary_seq = (data > median).astype(int)
    
    # 计算游程数
    runs_count = 1
    for i in range(1, len(binary_seq)):
        if binary_seq[i] != binary_seq[i-1]:
            runs_count += 1
    
    n1 = np.sum(binary_seq == 1)  # 高于中位数的个数
    n2 = np.sum(binary_seq == 0)  # 低于中位数的个数
    
    # 期望游程数
    expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
    
    # 游程数的方差
    var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    
    # Z统计量
    if var_runs > 0:
        z_stat = (runs_count - expected_runs) / np.sqrt(var_runs)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    else:
        z_stat = 0
        p_value = 1.0
    
    print(f"序列长度: {len(data)}")
    print(f"中位数: {median}")
    print(f"游程数: {runs_count}")
    print(f"期望游程数: {expected_runs:.2f}")
    print(f"Z统计量: {z_stat:.4f}")
    print(f"p值: {p_value:.6f}")
    
    conclusion = "通过" if p_value > 0.05 else "异常"
    print(f"\n结论: {conclusion} (α=0.05)")
    
    return z_stat, p_value, conclusion

def ks_test(data, name):
    """Kolmogorov-Smirnov检验"""
    print(f"\n{'=' * 60}")
    print(f"4. Kolmogorov-Smirnov检验 - {name}")
    print("=" * 60)
    
    data = np.array(data)
    
    # 均匀分布的CDF
    def uniform_cdf(x):
        return x / 10.0
    
    # 经验CDF
    sorted_data = np.sort(data)
    n = len(sorted_data)
    empirical_cdf = np.arange(1, n + 1) / n
    
    # KS统计量
    ks_stat = 0
    for x in range(10):
        # 计算理论CDF在x处的值
        theoretical = uniform_cdf(x + 1) - uniform_cdf(x)
        # 找到数据中在该区间的比例
        count = np.sum((sorted_data >= x) & (sorted_data <= x))
        observed_prop = count / n
        diff = abs(observed_prop - theoretical)
        ks_stat = max(ks_stat, diff)
    
    # 使用scipy的kstest
    ks_stat_scipy, p_value = kstest(data, 'uniform', args=(0, 10))
    
    print(f"序列长度: {len(data)}")
    print(f"KS统计量: {ks_stat_scipy:.4f}")
    print(f"p值: {p_value:.6f}")
    
    # 各数字频率与期望频率对比
    observed_freq = pd.Series(data).value_counts().sort_index() / len(data)
    expected_freq = 0.1
    print("\n各数字频率与期望(10%)对比:")
    abnormal = []
    for i in range(10):
        freq = observed_freq.get(i, 0)
        deviation = (freq - expected_freq) / expected_freq * 100
        status = ""
        if abs(deviation) > 5:
            status = "【异常】"
            abnormal.append(i)
        elif abs(deviation) > 3:
            status = "【可疑】"
        print(f"  数字{i}: 频率={freq*100:.2f}%, 偏差={deviation:+.2f}% {status}")
    
    conclusion = "通过" if p_value > 0.05 else "异常"
    if abnormal:
        conclusion = "异常"
    print(f"\n结论: {conclusion} (α=0.05)")
    
    return ks_stat_scipy, p_value, conclusion, abnormal

def autocorrelation_test(df, name, digit_col):
    """序列自相关检验"""
    print(f"\n{'=' * 60}")
    print(f"5. 序列自相关检验 - {name}")
    print("=" * 60)
    
    data = df[digit_col].values
    issues = df['issue'].values
    
    # 计算期号与数字的相关系数
    corr, p_value = stats.pearsonr(issues, data)
    
    print(f"序列长度: {len(data)}")
    print(f"皮尔逊相关系数: {corr:.6f}")
    print(f"p值: {p_value:.6f}")
    
    # 自相关分析 - 滞后1期
    data_centered = data - np.mean(data)
    n = len(data)
    lag1_corr = np.sum(data_centered[:-1] * data_centered[1:]) / (n * np.var(data))
    
    # 计算滞后1期自相关的置信区间
    ci = 1.96 / np.sqrt(n)
    
    print(f"滞后1期自相关系数: {lag1_corr:.6f}")
    print(f"95%置信区间: ±{ci:.6f}")
    
    conclusion = "通过" if abs(lag1_corr) < ci and p_value > 0.05 else "异常"
    print(f"\n结论: {conclusion} (α=0.05)")
    
    return corr, p_value, lag1_corr, conclusion

def parity_test(data, name):
    """奇偶性序列检验"""
    print(f"\n{'=' * 60}")
    print(f"6. 奇偶性序列检验 - {name}")
    print("=" * 60)
    
    # 转换为奇偶性 (奇数=1, 偶数=0)
    parity = np.array(data) % 2
    
    # 统计奇偶个数
    odd_count = np.sum(parity == 1)
    even_count = np.sum(parity == 0)
    
    print(f"奇数个数: {odd_count} ({odd_count/len(data)*100:.2f}%)")
    print(f"偶数个数: {even_count} ({even_count/len(data)*100:.2f}%)")
    print(f"期望比例: 50% / 50%")
    
    # 卡方检验
    expected = len(data) / 2
    chi2_stat = ((odd_count - expected) ** 2 + (even_count - expected) ** 2) / expected
    p_value = 1 - chi2.cdf(chi2_stat, 1)
    
    print(f"\n卡方统计量: {chi2_stat:.4f}")
    print(f"p值: {p_value:.6f}")
    
    # 游程检验
    runs_count = 1
    for i in range(1, len(parity)):
        if parity[i] != parity[i-1]:
            runs_count += 1
    
    n1, n2 = odd_count, even_count
    expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
    var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    
    if var_runs > 0:
        z_stat = (runs_count - expected_runs) / np.sqrt(var_runs)
        runs_p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    else:
        z_stat = 0
        runs_p_value = 1.0
    
    print(f"游程数: {runs_count}")
    print(f"期望游程数: {expected_runs:.2f}")
    print(f"游程Z统计量: {z_stat:.4f}")
    print(f"游程p值: {runs_p_value:.6f}")
    
    conclusion = "通过" if p_value > 0.05 and runs_p_value > 0.05 else "异常"
    print(f"\n结论: {conclusion} (α=0.05)")
    
    return chi2_stat, p_value, runs_count, runs_p_value, conclusion

def sum_value_analysis(df):
    """和值序列分析"""
    print(f"\n{'=' * 60}")
    print("7. 和值序列分析")
    print("=" * 60)
    
    # 计算和值
    df['sum'] = df['hundred'] + df['ten'] + df['one']
    
    sums = df['sum'].values
    
    print(f"和值范围: {sums.min()} - {sums.max()}")
    print(f"和值均值: {sums.mean():.2f}")
    print(f"和值标准差: {sums.std():.2f}")
    
    # 和值理论分布 (0-27)
    # 理论均值 = 9 * 3 = 13.5 (每个数字0-9均匀)
    theoretical_mean = 13.5
    print(f"理论均值: {theoretical_mean}")
    
    # 和值分布频率
    sum_counts = pd.Series(sums).value_counts().sort_index()
    
    # 卡方检验 - 和值分布
    # 理论频率: 从3个均匀分布(0-9)相加得到的分布
    # 每个位置独立，和值分布为离散卷积
    from scipy.stats import rv_histogram
    
    # 理论和值分布 (0-27)
    # P(S=s) = number of ways to get sum s / 1000
    # For 3 digits each 0-9: distribution is symmetric around 13.5
    theoretical_counts = {}
    for s in range(28):
        count = 0
        for a in range(10):
            for b in range(10):
                c = s - a - b
                if 0 <= c <= 9:
                    count += 1
        theoretical_counts[s] = count
    
    total_theoretical = sum(theoretical_counts.values())
    
    # 观察频数
    observed_counts = [sum_counts.get(s, 0) for s in range(28)]
    expected_counts = [theoretical_counts[s] * len(df) / 1000 for s in range(28)]
    
    # 卡方检验 (合并小概率类别)
    chi2_stat = 0
    for i in range(28):
        if expected_counts[i] > 0:
            chi2_stat += ((observed_counts[i] - expected_counts[i]) ** 2) / expected_counts[i]
    
    df_ks = 26  # 合并后自由度
    p_value = 1 - chi2.cdf(chi2_stat, df_ks)
    
    print(f"\n和值分布卡方检验:")
    print(f"卡方统计量: {chi2_stat:.4f}")
    print(f"自由度: {df_ks}")
    print(f"p值: {p_value:.6f}")
    
    conclusion = "通过" if p_value > 0.05 else "异常"
    print(f"\n结论: {conclusion} (α=0.05)")
    
    # 找出异常和值
    print("\n和值频率异常检测 (>3%偏差):")
    abnormal_sums = []
    for s in range(28):
        if sum_counts.get(s, 0) > 0:
            obs = sum_counts.get(s, 0)
            exp = expected_counts[s]
            if exp > 0:
                deviation = (obs - exp) / exp * 100
                if abs(deviation) > 10:
                    abnormal_sums.append((s, obs, exp, deviation))
                    print(f"  和值{s}: 观察={obs}, 期望={exp:.1f}, 偏差={deviation:+.2f}% 【异常】")
    
    return sums, chi2_stat, p_value, conclusion, sum_counts, theoretical_counts

def gap_distribution_test(data, name):
    """遗漏值分布检验 - 检验每个数字的遗漏间隔是否符合几何分布"""
    print(f"\n{'=' * 60}")
    print(f"8. 遗漏值分布检验 - {name}")
    print("=" * 60)
    
    results = []
    abnormal_digits = []
    
    for digit in range(10):
        digit_data = np.array(data)
        
        # 计算该数字的遗漏间隔
        gaps = []
        current_gap = 0
        for d in digit_data:
            if d == digit:
                if current_gap > 0:
                    gaps.append(current_gap)
                current_gap = 0
            else:
                current_gap += 1
        # 最后一个遗漏间隔不计入（未完成）
        
        if len(gaps) < 5:
            results.append((digit, np.nan, np.nan, "数据不足"))
            continue
        
        gaps = np.array(gaps)
        
        # 几何分布MLE估计
        # 对于几何分布 p = 1 / (mean + 1)
        mean_gap = np.mean(gaps)
        p_hat = 1 / (mean_gap + 1)
        
        # Kolmogorov-Smirnov检验
        # 几何分布CDF: P(X <= k) = 1 - (1-p)^(k+1)
        def geom_cdf(k, p):
            return 1 - (1 - p) ** (k + 1)
        
        sorted_gaps = np.sort(gaps)
        n = len(sorted_gaps)
        ks_stat = 0
        for i, g in enumerate(sorted_gaps):
            # i/n是经验CDF
            theoretical = geom_cdf(g, p_hat)
            empirical = (i + 1) / n
            ks_stat = max(ks_stat, abs(empirical - theoretical))
        
        # p值近似
        p_value = np.exp(-2 * n * ks_stat ** 2)
        
        status = "通过" if p_value > 0.05 else "【异常】"
        if p_value <= 0.05:
            abnormal_digits.append(digit)
        
        results.append((digit, ks_stat, p_value, status))
        print(f"  数字{digit}: 遗漏数={len(gaps)}, 平均遗漏={mean_gap:.2f}, KS={ks_stat:.4f}, p={p_value:.4f} {status}")
    
    total_abnormal = len(abnormal_digits)
    conclusion = "通过" if total_abnormal <= 1 else "异常"
    print(f"\n异常数字: {abnormal_digits if abnormal_digits else '无'}")
    print(f"结论: {conclusion} (α=0.05)")
    
    return results, abnormal_digits, conclusion

def plot_frequency(data, name, filename):
    """绘制频率分布图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 频率分布直方图
    counts = pd.Series(data).value_counts().sort_index()
    all_digits = pd.Series(index=range(10), data=0)
    for k, v in counts.items():
        all_digits[k] = v
    all_digits = all_digits.fillna(0)
    
    ax1 = axes[0]
    bars = ax1.bar(all_digits.index, all_digits.values, color='steelblue', alpha=0.7)
    ax1.axhline(y=len(data)/10, color='red', linestyle='--', label=f'期望频率 ({len(data)/10:.0f})')
    ax1.set_xlabel('数字')
    ax1.set_ylabel('出现次数')
    ax1.set_title(f'{name} - 数字频率分布')
    ax1.legend()
    
    # 偏差百分比
    ax2 = axes[1]
    expected = len(data) / 10
    deviations = (all_digits.values - expected) / expected * 100
    colors = ['red' if abs(d) > 5 else 'orange' if abs(d) > 3 else 'green' for d in deviations]
    bars2 = ax2.bar(all_digits.index, deviations, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axhline(y=5, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(y=-5, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('数字')
    ax2.set_ylabel('偏差百分比 (%)')
    ax2.set_title(f'{name} - 各数字偏差 (红色>5%, 橙色>3%)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {filename}")

def plot_sum_distribution(sum_counts, theoretical_counts, filename):
    """绘制和值分布图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sums = range(28)
    observed = [sum_counts.get(s, 0) for s in sums]
    expected = [theoretical_counts[s] for s in sums]
    
    # 分布对比
    ax1 = axes[0]
    x = np.arange(28)
    width = 0.35
    ax1.bar(x - width/2, observed, width, label='观察值', alpha=0.7)
    ax1.bar(x + width/2, expected, width, label='期望值', alpha=0.7)
    ax1.set_xlabel('和值')
    ax1.set_ylabel('频数')
    ax1.set_title('和值分布对比')
    ax1.legend()
    ax1.set_xticks(x)
    
    # 偏差百分比
    ax2 = axes[1]
    deviations = [(observed[i] - expected[i]) / expected[i] * 100 if expected[i] > 0 else 0 for i in range(28)]
    colors = ['red' if abs(d) > 10 else 'orange' if abs(d) > 5 else 'green' for d in deviations]
    ax2.bar(sums, deviations, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('和值')
    ax2.set_ylabel('偏差百分比 (%)')
    ax2.set_title('和值偏差百分比')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {filename}")

def plot_time_series(data, name, filename):
    """绘制时间序列图"""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(data, alpha=0.6, linewidth=0.5)
    ax.axhline(y=np.mean(data), color='red', linestyle='--', label=f'均值 ({np.mean(data):.2f})')
    ax.set_xlabel('期号')
    ax.set_ylabel('数字')
    ax.set_title(f'{name} - 时间序列')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {filename}")

def plot_parity(data, name, filename):
    """绘制奇偶性时序图"""
    parity = np.array(data) % 2
    
    fig, ax = plt.subplots(figsize=(14, 4))
    colors = ['blue' if p == 0 else 'red' for p in parity]
    ax.scatter(range(len(parity)), parity, c=colors, alpha=0.3, s=1)
    ax.set_xlabel('期号')
    ax.set_ylabel('奇偶性 (0=偶, 1=奇)')
    ax.set_title(f'{name} - 奇偶性序列')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['偶', '奇'])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {filename}")

def plot_correlation(data, issue_num, name, filename):
    """绘制期号与数字相关性图"""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.scatter(issue_num, data, alpha=0.3, s=2)
    
    # 添加趋势线
    z = np.polyfit(issue_num, data, 1)
    p = np.poly1d(z)
    ax.plot(issue_num, p(issue_num), "r--", alpha=0.8, label=f'趋势线 (斜率={z[0]:.6f})')
    
    ax.set_xlabel('期号')
    ax.set_ylabel('数字')
    ax.set_title(f'{name} - 期号与数字相关性')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {filename}")

def plot_gap_histogram(data, digit, name, filename):
    """绘制遗漏间隔直方图"""
    gaps = []
    current_gap = 0
    for d in data:
        if d == digit:
            if current_gap > 0:
                gaps.append(current_gap)
            current_gap = 0
        else:
            current_gap += 1
    
    if len(gaps) < 5:
        return
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(gaps, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    
    # 叠加几何分布曲线
    mean_gap = np.mean(gaps)
    p_hat = 1 / (mean_gap + 1)
    x = np.arange(1, max(gaps) + 1)
    geom_prob = (1 - p_hat) ** (x - 1) * p_hat
    ax.plot(x, geom_prob, 'r-', linewidth=2, label=f'几何分布 (p={p_hat:.4f})')
    
    ax.set_xlabel('遗漏间隔')
    ax.set_ylabel('密度')
    ax.set_title(f'{name} - 数字{digit} 遗漏间隔分布')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {filename}")

def main():
    print("\n" + "=" * 70)
    print("中国福利彩票3D PRNG统计检验报告")
    print("=" * 70)
    
    # 加载数据
    df = load_data()
    
    # 提取三个位置的数字序列
    hundred = df['hundred'].values
    ten = df['ten'].values
    one = df['one'].values
    all_digits = np.concatenate([hundred, ten, one])
    
    results_summary = []
    
    # 1. 频率统计
    print(f"\n总数据量: {len(all_digits)} (3个位置 × {len(df)}期)")
    
    # 2. 卡方检验
    chi2_h, p_h, con_h, ab_h = chi_square_test(hundred, "百位")
    chi2_t, p_t, con_t, ab_t = chi_square_test(ten, "十位")
    chi2_o, p_o, con_o, ab_o = chi_square_test(one, "个位")
    chi2_all, p_all, con_all, ab_all = chi_square_test(all_digits, "合并序列")
    
    results_summary.append(("卡方检验-百位", chi2_h, p_h, con_h))
    results_summary.append(("卡方检验-十位", chi2_t, p_t, con_t))
    results_summary.append(("卡方检验-个位", chi2_o, p_o, con_o))
    results_summary.append(("卡方检验-合并", chi2_all, p_all, con_all))
    
    # 3. 游程检验
    runs_h, p_rh, con_rh = runs_test(hundred, "百位")
    runs_t, p_rt, con_rt = runs_test(ten, "十位")
    runs_o, p_ro, con_ro = runs_test(one, "个位")
    runs_all, p_ra, con_ra = runs_test(all_digits, "合并序列")
    
    results_summary.append(("游程检验-百位", runs_h, p_rh, con_rh))
    results_summary.append(("游程检验-十位", runs_t, p_rt, con_rt))
    results_summary.append(("游程检验-个位", runs_o, p_ro, con_ro))
    results_summary.append(("游程检验-合并", runs_all, p_ra, con_ra))
    
    # 4. KS检验
    ks_h, p_kh, con_kh, ab_kh = ks_test(hundred, "百位")
    ks_t, p_kt, con_kt, ab_kt = ks_test(ten, "十位")
    ks_o, p_ko, con_ko, ab_ko = ks_test(one, "个位")
    ks_all, p_ka, con_ka, ab_ka = ks_test(all_digits, "合并序列")
    
    results_summary.append(("KS检验-百位", ks_h, p_kh, con_kh))
    results_summary.append(("KS检验-十位", ks_t, p_kt, con_kt))
    results_summary.append(("KS检验-个位", ks_o, p_ko, con_ko))
    results_summary.append(("KS检验-合并", ks_all, p_ka, con_ka))
    
    # 5. 自相关检验
    print("\n" + "=" * 60)
    print("5. 序列自相关检验")
    print("=" * 60)
    
    issue_nums = np.arange(1, len(df) + 1)
    
    corr_h, p_corr_h, lag1_h, con_corr_h = autocorrelation_test(df, "百位", 'hundred')
    corr_t, p_corr_t, lag1_t, con_corr_t = autocorrelation_test(df, "十位", 'ten')
    corr_o, p_corr_o, lag1_o, con_corr_o = autocorrelation_test(df, "个位", 'one')
    
    results_summary.append(("自相关-百位", corr_h, p_corr_h, con_corr_h))
    results_summary.append(("自相关-十位", corr_t, p_corr_t, con_corr_t))
    results_summary.append(("自相关-个位", corr_o, p_corr_o, con_corr_o))
    
    # 6. 奇偶性检验
    print("\n" + "=" * 60)
    print("6. 奇偶性序列检验")
    print("=" * 60)
    
    chi2_p_h, p_ph, runs_p_h, p_rph, con_ph = parity_test(hundred, "百位")
    chi2_p_t, p_pt, runs_p_t, p_rpt, con_pt = parity_test(ten, "十位")
    chi2_p_o, p_po, runs_p_o, p_rpo, con_po = parity_test(one, "个位")
    
    results_summary.append(("奇偶-百位", chi2_p_h, p_ph, con_ph))
    results_summary.append(("奇偶-十位", chi2_p_t, p_pt, con_pt))
    results_summary.append(("奇偶-个位", chi2_p_o, p_po, con_po))
    
    # 7. 和值分析
    sums, chi2_sum, p_sum, con_sum, sum_counts, theo_counts = sum_value_analysis(df)
    
    results_summary.append(("和值分布", chi2_sum, p_sum, con_sum))
    
    # 8. 遗漏值分布检验
    gap_h, ab_gap_h, con_gap_h = gap_distribution_test(hundred, "百位")
    gap_t, ab_gap_t, con_gap_t = gap_distribution_test(ten, "十位")
    gap_o, ab_gap_o, con_gap_o = gap_distribution_test(one, "个位")
    
    results_summary.append(("遗漏分布-百位", np.nan, np.nan, con_gap_h))
    results_summary.append(("遗漏分布-十位", np.nan, np.nan, con_gap_t))
    results_summary.append(("遗漏分布-个位", np.nan, np.nan, con_gap_o))
    
    # 生成可视化图表
    print("\n" + "=" * 60)
    print("生成可视化图表")
    print("=" * 60)
    
    plot_frequency(hundred, "百位", "hundred_frequency.png")
    plot_frequency(ten, "十位", "ten_frequency.png")
    plot_frequency(one, "个位", "one_frequency.png")
    plot_frequency(all_digits, "合并序列", "all_frequency.png")
    
    plot_sum_distribution(sum_counts, theo_counts, "sum_distribution.png")
    
    plot_time_series(hundred, "百位", "hundred_timeseries.png")
    plot_time_series(ten, "十位", "ten_timeseries.png")
    plot_time_series(one, "个位", "one_timeseries.png")
    
    plot_parity(hundred, "百位", "hundred_parity.png")
    plot_parity(ten, "十位", "ten_parity.png")
    plot_parity(one, "个位", "one_parity.png")
    
    plot_correlation(hundred, issue_nums, "百位", "hundred_correlation.png")
    plot_correlation(ten, issue_nums, "十位", "ten_correlation.png")
    plot_correlation(one, issue_nums, "个位", "one_correlation.png")
    
    # 遗漏间隔图
    for digit in range(10):
        plot_gap_histogram(hundred, digit, "百位", f"hundred_gap_{digit}.png")
        plot_gap_histogram(ten, digit, "十位", f"ten_gap_{digit}.png")
        plot_gap_histogram(one, digit, "个位", f"one_gap_{digit}.png")
    
    # 打印最终总结
    print("\n" + "=" * 70)
    print("检验结果汇总")
    print("=" * 70)
    print(f"{'检验项目':<25} {'统计量':<15} {'p值':<15} {'结论':<10}")
    print("-" * 70)
    for name, stat, pval, con in results_summary:
        stat_str = f"{stat:.4f}" if not np.isnan(stat) else "N/A"
        pval_str = f"{pval:.6f}" if not np.isnan(pval) else "N/A"
        print(f"{name:<25} {stat_str:<15} {pval_str:<15} {con:<10}")
    
    # 统计异常项
    abnormal_count = sum(1 for _, _, _, c in results_summary if c == "异常")
    suspicious_count = sum(1 for _, _, _, c in results_summary if c == "可疑")
    
    print("\n" + "=" * 70)
    print("最终结论")
    print("=" * 70)
    print(f"总检验项数: {len(results_summary)}")
    print(f"异常项数: {abnormal_count}")
    print(f"可疑项数: {suspicious_count}")
    
    if abnormal_count == 0 and suspicious_count == 0:
        print("\n该彩票序列通过所有统计检验，未发现明显的随机性异常。")
    elif abnormal_count <= 2:
        print(f"\n发现{abnormal_count}项异常，建议关注但不能确定存在问题。")
    else:
        print(f"\n发现{abnormal_count}项异常和{suspicious_count}项可疑，序列可能存在非随机特征。")
    
    print("\n详细图表已保存至:", OUTPUT_DIR)
    
    # 保存报告到文件
    report_path = os.path.join(OUTPUT_DIR, 'prng_test_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("中国福利彩票3D PRNG统计检验报告\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"数据量: {len(df)}期\n")
        f.write(f"时间范围: {df['date'].iloc[0]} 至 {df['date'].iloc[-1]}\n\n")
        f.write("检验结果汇总\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'检验项目':<25} {'统计量':<15} {'p值':<15} {'结论':<10}\n")
        for name, stat, pval, con in results_summary:
            stat_str = f"{stat:.4f}" if not np.isnan(stat) else "N/A"
            pval_str = f"{pval:.6f}" if not np.isnan(pval) else "N/A"
            f.write(f"{name:<25} {stat_str:<15} {pval_str:<15} {con:<10}\n")
        f.write("\n最终结论\n")
        f.write(f"异常项数: {abnormal_count}\n")
        f.write(f"可疑项数: {suspicious_count}\n")
    
    print(f"文本报告已保存: {report_path}")

if __name__ == "__main__":
    main()