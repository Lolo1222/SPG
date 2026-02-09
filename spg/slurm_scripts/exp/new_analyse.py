import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# ==================== 论文绘图专用配置 ====================
# 1. 设置字体为 Times New Roman (论文标准)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
# 2. 设置数学公式字体风格 (接近 LaTeX)
plt.rcParams['mathtext.fontset'] = 'stix'
# 3. 设置字号和线条粗细
sns.set_context("paper", font_scale=1.8) # 增大字号，确保缩小后依然清晰
sns.set_style("whitegrid", {
    "grid.linestyle": "--", 
    "grid.alpha": 0.5,
    "axes.edgecolor": "black",
    "axes.linewidth": 1.2
})
# ========================================================

def load_data(file_path, label):
    """
    加载 jsonl 日志文件并转换为 DataFrame。
    """
    if not os.path.exists(file_path):
        print(f"警告: 文件未找到 - {file_path}")
        return None

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                line = line.strip()
                if not line: continue
                
                record = json.loads(line)
                
                var = record.get('elbo_variance', None)
                step = record.get('step', None)
                fixed_ratio = record.get('fixed_ratio', 0.0)
                
                if var is not None and step is not None:
                    data.append({
                        'Step': int(step),
                        'Variance': float(var),
                        'Fixed Ratio': float(fixed_ratio),
                        'Method': label
                    })
            except Exception:
                continue
    return pd.DataFrame(data)

# ================= 配置区 =================
# 请确保这两个文件名与您训练生成的日志文件名一致
standard_file = "save_dir/run_results/exp_math_elbo_grpo_num_t3_20260126_014235/elbo_variance_stats.jsonl"       # Baseline (Standard) 日志
semi_offline_file = "save_dir/run_results/exp_math_swift_grpo_generated_num_t3_semi0.95_mask_answer_low_confidence_early0.95_20260126_014257/semi_offline_variance.jsonl" # Ours (Semi-Offline) 日志
# ==========================================

# 1. 加载数据
print("正在加载数据...")
df_std = load_data(standard_file, "GRPO w/ ELBO")
df_our = load_data(semi_offline_file, "STP (Ours)")

df_list = []
if df_std is not None and not df_std.empty:
    df_list.append(df_std)
if df_our is not None and not df_our.empty:
    df_list.append(df_our)

if not df_list:
    raise ValueError("错误：没有有效数据。")

df = pd.concat(df_list, ignore_index=True)

# --- 图表 1: Variance Over Steps (折线图) ---
plt.figure(figsize=(8, 6)) # 论文通常使用 4:3 或方形比例

# 定义颜色板，使用高对比度颜色（适合黑白打印区分）
# palette = ["#E24A33", "#348ABD"] # 经典的学术红蓝配色
palette = {
    "GRPO w/ ELBO": "tab:blue",           # 经典蓝
    "STP (Ours)": "tab:orange" # 经典橙
}
sns.lineplot(
    data=df, 
    x='Step', 
    y='Variance', 
    hue='Method', 
    style='Method', 
    markers=True, 
    dashes=False, 
    err_style='band',
    palette=palette,
    linewidth=2.5,  # 加粗线条
    markersize=8    # 加大标记点
)

plt.title("ELBO Estimator Variance During Training", fontsize=18, fontweight='bold', pad=15)
plt.ylabel(r"ELBO Variance (log scale)", fontsize=16) # 使用 LaTeX 格式
plt.xlabel("Training Steps", fontsize=16)
plt.yscale("log") 
plt.legend(frameon=True, fontsize=12, loc='best') # 显示图例边框
plt.grid(True, which="both", ls="--", alpha=0.3)

# 保存为 PDF
plt.savefig("figures/newnew/fig1_variance_steps.pdf", format='pdf', bbox_inches='tight', dpi=300)
print("图表已保存: fig1_variance_steps.pdf")
plt.show()

# --- 图表 2: Variance Distribution (箱线图) ---
plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df, 
    x='Method', 
    y='Variance', 
    palette=palette, 
    showfliers=False,
    width=0.5,
    linewidth=1.5 # 加粗箱线边框
)
sns.stripplot(
    data=df, 
    x='Method', 
    y='Variance', 
    color=".2", 
    alpha=0.4, 
    size=4,
    jitter=True
)

plt.title("Overall Distribution of ELBO Estimation Variance", fontsize=18, fontweight='bold', pad=15)
plt.ylabel("ELBO Variance", fontsize=16)
plt.xlabel("Method") # 移除多余的 X 轴标签
plt.xticks(fontsize=14)

plt.savefig("figures/newnew/fig2_variance_dist.pdf", format='pdf', bbox_inches='tight', dpi=300)
print("图表已保存: fig2_variance_dist.pdf")
plt.show()

