#!/usr/bin/env bash
set -euo pipefail

# wrapper that saves stdout/stderr to timestamped log file under logs/
LOGDIR="logs"
mkdir -p "$LOGDIR"
TS=$(date +"%Y%m%d_%H%M%S")
LOG="$LOGDIR/unmask_rollout_${TS}.log"

echo "Writing unmask_rollout logs to $LOG"

python /home/jwliu/dlm/SPG/scripts/unmask_rollout.py \
  --jsonl deal_data/math02/llada_math_generations_rank0.jsonl \
  --model_path /home/jwliu/dlm/SPG/save_dir/hf_models/LLaDA-8B-Instruct \
  --p_gen_mask 0.3 \
  --mask_strategy lowprob \
  --rollout_nums 5 \
  --temperature 1.0 \
  --gpus 0 \
  > "$LOG" 2>&1

echo "Run finished. Log saved to $LOG"