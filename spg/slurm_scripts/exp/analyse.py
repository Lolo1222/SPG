import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

def load_data(file_path, label):
    """
    加载 jsonl 日志文件并转换为 DataFrame。
    如果文件不存在，返回 None。
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
                
                # 提取关键字段，设置默认值以防字段缺失
                var = record.get('elbo_variance', None)
                step = record.get('step', None)
                
                # Standard 方法可能没有 fixed_ratio，默认为 0
                fixed_ratio = record.get('fixed_ratio', 0.0)
                
                # 只有当关键数据存在时才添加
                if var is not None and step is not None:
                    data.append({
                        'Step': int(step),
                        'Variance': float(var),
                        'Fixed Ratio': float(fixed_ratio),
                        'Method': label
                    })
            except json.JSONDecodeError:
                print(f"警告: {file_path} 第 {line_num+1} 行 JSON 解析失败，跳过。")
                continue
            except Exception as e:
                print(f"警告: 处理 {file_path} 时出错: {e}")
                continue
                
    print(f"成功从 {file_path} 加载了 {len(data)} 条数据。")
    return pd.DataFrame(data)

# ================= 配置区 =================
# 请确保这两个文件名与您训练生成的日志文件名一致
standard_file = "save_dir/run_results/exp_math_elbo_grpo_num_t3_20260126_014235/elbo_variance_stats.jsonl"       # Baseline (Standard) 日志
semi_offline_file = "save_dir/run_results/exp_math_swift_grpo_generated_num_t3_semi0.95_mask_answer_low_confidence_early0.95_20260126_014257/semi_offline_variance.jsonl" # Ours (Semi-Offline) 日志
# ==========================================

# 1. 加载数据
print("正在加载数据...")
df_std = load_data(standard_file, "Standard")
df_our = load_data(semi_offline_file, "Ours (Semi-Offline)")

# 2. 合并数据
if df_std is None and df_our is None:
    raise FileNotFoundError("错误：未找到任何日志文件。请先运行训练脚本生成 .jsonl 文件。")

df_list = []
if df_std is not None and not df_std.empty:
    df_list.append(df_std)
if df_our is not None and not df_our.empty:
    df_list.append(df_our)

if not df_list:
    raise ValueError("错误：日志文件存在但没有有效数据。")

df = pd.concat(df_list, ignore_index=True)

# 3. 设置绘图风格
sns.set(style="whitegrid", context="paper", font_scale=1.4)

# --- 图表 1: Variance Over Steps (折线图) ---
# 聚合：计算每个 Step 的平均方差（因为每个 Step 有多个样本）
plt.figure(figsize=(12, 7))

# lineplot 会自动计算置信区间 (默认 95% CI)，显示的带状区域即代表了数据的波动范围
sns.lineplot(
    data=df, 
    x='Step', 
    y='Variance', 
    hue='Method', 
    style='Method', 
    markers=True, 
    dashes=False, 
    err_style='band',  # 显示误差带
    errorbar=('ci', 95)            # 95% 置信区间
)

plt.title("ELBO Estimator Variance During Training")
plt.ylabel("ELBO Variance (Log Scale)")
plt.xlabel("Training Steps")
plt.yscale("log") # 强烈建议使用对数坐标，因为方差差异可能在数量级上
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig("figures/variance_comparison_steps.pdf", format='pdf', dpi=300)
print("图表已保存: figures/variance_comparison_steps.pdf")
plt.show()

# --- 图表 2: Variance Distribution (箱线图) ---
# 展示整体分布情况，验证 Semi-offline 是否有更少的离群值
plt.figure(figsize=(10, 7))

# 排除极端的离群值以便看清主体分布
sns.boxplot(
    data=df, 
    x='Method', 
    y='Variance', 
    palette="Set2", 
    showfliers=False # 不显示极端离群点
)
# 叠加抖动散点图，展示真实数据密度
sns.stripplot(
    data=df, 
    x='Method', 
    y='Variance', 
    color=".2", 
    alpha=0.3, 
    size=3,
    jitter=True
)

plt.title("Overall Distribution of ELBO Variance")
plt.ylabel("ELBO Variance")
plt.tight_layout()
plt.savefig("figures/variance_distribution_boxplot.pdf", format='pdf', dpi=300)
print("图表已保存: figures/variance_distribution_boxplot.pdf")
plt.show()

# --- 图表 3: Variance vs Fixed Ratio (仅针对 Ours) ---
# 验证定理：Fixed Ratio 越高，方差是否越低
# if df_our is not None and not df_our.empty:
#     # 过滤掉 Fixed Ratio 为 0 的数据（如果有的话）
#     df_analysis = df_our[df_our['Fixed Ratio'] > 0]
    
#     if not df_analysis.empty:
#         plt.figure(figsize=(10, 7))
        
#         # 散点图
#         sns.scatterplot(
#             data=df_analysis, 
#             x='Fixed Ratio', 
#             y='Variance', 
#             alpha=0.6,
#             s=80,
#             edgecolor='w'
#         )
        
#         # 尝试拟合回归线
#         try:
#             sns.regplot(
#                 data=df_analysis, 
#                 x='Fixed Ratio', 
#                 y='Variance', 
#                 scatter=False, 
#                 color='red', 
#                 line_kws={'linestyle': '--', 'label': 'Trend'}
#             )
#             plt.legend()
#         except Exception:
#             pass # 如果数据点太少无法回归则跳过

#         plt.title("Effect of Fixed Token Ratio on Variance (Semi-Offline)")
#         plt.xlabel("Fixed Token Ratio (ρ)")
#         plt.ylabel("ELBO Variance")
#         plt.tight_layout()
#         plt.savefig("figures/variance_vs_fixed_ratio.png", dpi=300)
#         print("图表已保存: figures/variance_vs_fixed_ratio.png")
#         plt.show()
#     else:
#         print("提示: Semi-Offline 数据中没有检测到 Fixed Ratio > 0 的样本，跳过图表 3。")