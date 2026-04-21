#!/usr/bin/env bash
set -euo pipefail

python sampling/compare_shift_results.py \
	--summary \
	"sampling/results/qwen3_1p7b_vs_4b_gsm8k_ns1024/run_summary.json" \
	"sampling/results/qwen3_4b_vs_1p7b_gsm8k_ns1024/run_summary.json" \
	"sampling/results/qwen3_4b_vs_4b_gsm8k_ns1024/run_summary.json" \
	"sampling/results/llada_8b_vs_qwen3_4b_gsm8k_bs1_ns1024/run_summary.json" \
	"sampling/results/llada_8b_vs_qwen3_4b_gsm8k_bs2_ns1024/run_summary.json" \
	"sampling/results/llada_8b_vs_qwen3_4b_gsm8k_bs4_ns1024/run_summary.json" \
	"sampling/results/llada_8b_vs_qwen3_4b_gsm8k_bs8_ns1024/run_summary.json"