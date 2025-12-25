#!/usr/bin/env python3
# scripts/merge_by_question_random_file.py
import json
import random
import os
from glob import glob

# Config
input_paths = []
# input_paths += glob("deal_data/math01/*.jsonl")
input_paths += glob("deal_data/math01/llada_math_generations.jsonl")
input_paths += glob("deal_data/math02/llada_math_generations_rank*.jsonl")
output_jsonl = "deal_data/math_merged.jsonl"
output_summary = "deal_data/math_merged_summary.json"
seed = 42  # 若需可设置为整数以复现，例如 seed = 42

def main():
    files = list(input_paths)
    if not files:
        print("没有找到输入文件。请确认路径下存在 `deal_data/math01` 和 `deal_data/math02` 的 jsonl 文件。")
        return

    if seed is not None:
        random.seed(seed)
    random.shuffle(files)

    seen = set()
    num_samples = 0
    best_of_N_correct_count = 0
    per_file_counts = {}

    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for fpath in files:
            basename = os.path.basename(fpath)
            try:
                with open(fpath, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception as e:
                            # 跳过无法解析的行（但报告）
                            print(f"警告：无法解析 {fpath} 的一行：{e}")
                            continue
                        q = obj.get("question")
                        if q is None:
                            # 如果没有 question 字段，跳过
                            continue
                        if q in seen:
                            continue
                        # First occurrence (because files order is randomized)
                        seen.add(q)
                        out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                        num_samples += 1
                        if obj.get("best_of_N_correct") is True:
                            best_of_N_correct_count += 1
                        per_file_counts[basename] = per_file_counts.get(basename, 0) + 1
            except FileNotFoundError:
                print(f"警告：文件未找到 {fpath}, 跳过。")
            except Exception as e:
                print(f"警告：读取文件 {fpath} 出错：{e}")

    best_of_N_accuracy = (best_of_N_correct_count / num_samples) if num_samples > 0 else 0.0

    summary = {
        "num_samples": num_samples,
        "best_of_N_correct": best_of_N_correct_count,
        "best_of_N_accuracy": best_of_N_accuracy,
        "per_file_counts": per_file_counts,
        "input_files": files,
    }

    with open(output_summary, "w", encoding="utf-8") as s_f:
        json.dump(summary, s_f, ensure_ascii=False, indent=2)

    print("合并完成。")
    print(f"合并文件：{output_jsonl}")
    print(f"汇总文件：{output_summary}")
    print(f"样本数：{num_samples}, best_of_N_correct={best_of_N_correct_count}, accuracy={best_of_N_accuracy:.6f}")

if __name__ == "__main__":
    main()