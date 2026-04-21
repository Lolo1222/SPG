#!/usr/bin/env python3
"""Compare shift-analysis run summaries across experiments.

This script is designed for horizontal comparison between:
- diffusion vs llm runs (from diffusion_llm_shift_analysis.py)
- llm vs llm runs (from llm_llm_shift_analysis.py)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_path(d: Dict[str, Any], keys: List[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def detect_run_type(summary: Dict[str, Any]) -> str:
    cfg = summary.get("config", {})
    if "diffusion_model_path" in cfg:
        return "diffusion_vs_llm"
    if "model_a_path" in cfg and "model_b_path" in cfg:
        return "llm_vs_llm"
    return "unknown"


def extract_core_metrics(name: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    run_type = detect_run_type(summary)
    cfg = summary.get("config", {})
    ess = summary.get("ess", {})

    row: Dict[str, Any] = {
        "run_name": name,
        "run_type": run_type,
        "dataset": cfg.get("dataset"),
        "num_samples": cfg.get("num_samples"),
    }

    if run_type == "diffusion_vs_llm":
        row["source_model"] = cfg.get("diffusion_model_path")
        row["target_model"] = cfg.get("llm_model_path")

        row["seq_ess_ratio_source_over_target"] = get_path(
            ess, ["sequence_level_diffusion_over_llm", "ess_ratio"]
        )
        row["seq_ess_ratio_target_over_source"] = get_path(
            ess, ["sequence_level_llm_over_diffusion", "ess_ratio"]
        )
        row["tok_ess_ratio_source_over_target"] = get_path(
            ess, ["token_level_diffusion_over_llm", "ess_ratio"]
        )
        row["tok_ess_ratio_target_over_source"] = get_path(
            ess, ["token_level_llm_over_diffusion", "ess_ratio"]
        )
        row["blk_ess_ratio_source_over_target"] = get_path(
            ess, ["block_level_diffusion_over_llm", "ess_ratio"]
        )
        row["blk_ess_ratio_target_over_source"] = get_path(
            ess, ["block_level_llm_over_diffusion", "ess_ratio"]
        )

        row["mean_abs_gap"] = get_path(summary, ["sample_summary", "mean_abs_total_gap"])
        row["pearson_total"] = get_path(summary, ["sample_summary", "pearson_total_diff_vs_llm"])
        row["spearman_total"] = get_path(summary, ["sample_summary", "spearman_total_diff_vs_llm"])
        row["token_aligned_samples"] = get_path(ess, ["token_alignment", "aligned_samples"])

    elif run_type == "llm_vs_llm":
        row["source_model"] = cfg.get("model_a_path")
        row["target_model"] = cfg.get("model_b_path")

        row["seq_ess_ratio_source_over_target"] = get_path(
            ess, ["sequence_level_a_over_b", "ess_ratio"]
        )
        row["seq_ess_ratio_target_over_source"] = get_path(
            ess, ["sequence_level_b_over_a", "ess_ratio"]
        )
        row["tok_ess_ratio_source_over_target"] = get_path(
            ess, ["token_level_a_over_b", "ess_ratio"]
        )
        row["tok_ess_ratio_target_over_source"] = get_path(
            ess, ["token_level_b_over_a", "ess_ratio"]
        )
        row["blk_ess_ratio_source_over_target"] = None
        row["blk_ess_ratio_target_over_source"] = None

        row["mean_abs_gap"] = None
        row["pearson_total"] = None
        row["spearman_total"] = None
        row["token_aligned_samples"] = get_path(ess, ["token_level_a_over_b", "n_samples"])

    else:
        row["source_model"] = None
        row["target_model"] = None

    row["trainability_hint"] = assess_trainability(row)
    return row


def assess_trainability(row: Dict[str, Any]) -> str:
    seq_ratio = row.get("seq_ess_ratio_target_over_source")
    tok_ratio = row.get("tok_ess_ratio_target_over_source")

    if seq_ratio is None or tok_ratio is None:
        return "unknown"
    if seq_ratio >= 0.5 and tok_ratio >= 0.5:
        return "good"
    if seq_ratio >= 0.2 and tok_ratio >= 0.3:
        return "medium"
    return "hard"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]

    for _, row in df.iterrows():
        values = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                values.append("")
            else:
                values.append(str(v))
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple shift-analysis run summaries")
    parser.add_argument(
        "--summary",
        type=Path,
        nargs="+",
        required=True,
        help="One or more run_summary.json paths",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("sampling/results/comparisons"))
    parser.add_argument("--run_name", type=str, default="compare_shift_runs")
    args = parser.parse_args()

    out_dir = args.output_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for p in args.summary:
        summary = load_json(p)
        run_name = p.parent.name
        rows.append(extract_core_metrics(run_name, summary))

    df = pd.DataFrame(rows)
    csv_path = out_dir / "comparison.csv"
    json_path = out_dir / "comparison.json"
    md_path = out_dir / "comparison.md"

    df.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Shift Comparison\n\n")
        f.write(dataframe_to_markdown(df))
        f.write("\n")

    print(f"Wrote comparison CSV: {csv_path}")
    print(f"Wrote comparison JSON: {json_path}")
    print(f"Wrote comparison Markdown: {md_path}")


if __name__ == "__main__":
    main()
