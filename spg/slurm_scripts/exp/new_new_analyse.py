import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# ==================== 论文绘图专用配置 ====================
# 1. 设置基础风格：Whitegrid
sns.set_theme(style="whitegrid", context="paper", font_scale=2.0)

# 2. 字体设置：Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'Liberation Serif', 'serif']
plt.rcParams['mathtext.fontset'] = 'stix'

# 3. 核心配色：经典蓝橙 (Blue & Orange)
# 这种配色对比度高，且符合大多数人的视觉习惯
palette = {
    "Standard": "tab:blue",           # 经典蓝
    "Ours (Semi-Offline)": "tab:orange" # 经典橙
}
# ========================================================

def load_data(file_path, label):
    if not os.path.exists(file_path):
        return None
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                line = line.strip()
                if not line: continue
                record = json.loads(line)
                var = record.get('elbo_variance', None)
                step = record.get('step', None)
                if var is not None and step is not None:
                    data.append({
                        'Step': int(step),
                        'Variance': float(var),
                        'Method': label
                    })
            except: continue
    return pd.DataFrame(data)


# ================= 配置区 =================
# 请确保这两个文件名与您训练生成的日志文件名一致
standard_file = "save_dir/run_results/exp_math_elbo_grpo_num_t3_20260126_014235/elbo_variance_stats.jsonl"       # Baseline (Standard) 日志
semi_offline_file = "save_dir/run_results/exp_math_swift_grpo_generated_num_t3_semi0.95_mask_answer_low_confidence_early0.95_20260126_014257/semi_offline_variance.jsonl" # Ours (Semi-Offline) 日志
# ==========================================

# 加载数据
print("正在加载数据...")
df_std = load_data(standard_file, "Standard")
df_our = load_data(semi_offline_file, "Ours (Semi-Offline)")
df = pd.concat([d for d in [df_std, df_our] if d is not None and not d.empty], ignore_index=True)

# -----------------------------------------------------------
# 图表 2: Variance Distribution (Blue & Orange 风格)
# -----------------------------------------------------------
plt.figure(figsize=(7, 6)) # 调整为适合单栏插图的尺寸

# 1. 绘制箱线图 (Boxplot)
# saturation=0.75 让颜色稍微柔和一点，boxprops alpha=0.6 让填充色半透明
ax = sns.boxplot(
    data=df, 
    x='Method', 
    y='Variance', 
    palette=palette,
    showfliers=False,       # 隐藏极端异常值(由散点展示)
    width=0.5,              # 箱体宽度适中
    linewidth=2.0,          # 边框线条加粗
    saturation=0.85,        # 颜色饱和度
    boxprops=dict(alpha=0.7) # 填充透明度
)

# 2. 绘制散点图 (Stripplot)
# 颜色设为深灰色或黑色，叠加在箱体上，增加数据的真实感
sns.stripplot(
    data=df, 
    x='Method', 
    y='Variance', 
    color="#333333",        # 使用深灰色点，避免颜色干扰
    alpha=0.4,              # 半透明，避免遮挡
    size=5,                 # 点的大小
    jitter=True,            # 随机抖动
    ax=ax
)

# 3. 细节美化
ax.set_title("Distribution of ELBO Variance", fontsize=22, fontweight='bold', pad=20)
ax.set_ylabel("ELBO Variance", fontsize=20)
ax.set_xlabel("") # 移除X轴标签，因为图例/刻度已经说明了
plt.xticks(fontsize=18)
plt.yticks(fontsize=16)

# 移除顶部和右侧的边框 (Academic Style)
sns.despine(trim=True, offset=10)

# 调整网格线
ax.grid(True, axis='y', linestyle='--', alpha=0.5, linewidth=1.0)

# 保存
plt.tight_layout()
plt.savefig("figures/new/fig2_variance_dist_blue_orange.pdf", format='pdf', bbox_inches='tight', dpi=300)
print("图表已保存: fig2_variance_dist_blue_orange.pdf")
plt.show()