#!/usr/bin/env bash
set -euo pipefail

# Re-run the paper's training jobs one at a time.  This script intentionally
# does not submit jobs to Slurm and defaults to a dry run.  Use RUN=1 only
# after checking the printed command and GPU ownership.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${GPU_ID:-3}"
RUN="${RUN:-0}"
CONDA_ENV="${CONDA_ENV:-spg}"
MODEL_PATH="${MODEL_PATH:-/root/Models/LLaDA-8B-Instruct}"
TARGET_EFFECTIVE_BATCH="${TARGET_EFFECTIVE_BATCH:-12}"
NUM_GENERATIONS="${NUM_GENERATIONS:-6}"
BASE_GRAD_ACCUM="${BASE_GRAD_ACCUM:-2}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required for the GPU ownership check" >&2
  exit 1
fi
gpu_count=$(awk -F',' '{print NF}' <<< "$GPU_ID")
if (( TARGET_EFFECTIVE_BATCH < 1 || NUM_GENERATIONS < 2 )); then
  echo "TARGET_EFFECTIVE_BATCH and NUM_GENERATIONS must be positive" >&2
  exit 1
fi

found_gpu=0
while IFS= read -r gpu_row; do
  gpu_index=${gpu_row%%,*}; gpu_used=${gpu_row##*,}
  gpu_index=${gpu_index// /}; gpu_used=${gpu_used// /}
  case ",$GPU_ID," in *,"$gpu_index",*) ;; *) continue ;; esac
  found_gpu=1
  if (( gpu_used > ${GPU_FREE_THRESHOLD_MB:-1024} )); then
    echo "GPU $gpu_index is occupied (${gpu_used} MiB used); refusing to start." >&2
    exit 2
  fi
done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
if (( found_gpu == 0 )); then
  echo "One or more selected GPUs were not found: GPU_ID=$GPU_ID" >&2
  exit 1
fi

# Preserve the one-GPU reference update size: 6 samples per dataloader batch
# and accumulation 2 => 12 samples per optimizer update. For each visible GPU
# count, find an integer per-device batch/accumulation pair with the same
# effective global batch. This keeps the optimizer step schedule and learning
# rate semantics unchanged. Exact preservation is possible when the selected
# GPU count divides the target batch; otherwise fail instead of silently
# changing the experiment.
PER_DEVICE_TRAIN_BATCH_SIZE=""
GRADIENT_ACCUMULATION_STEPS=""
for candidate_ga in "$BASE_GRAD_ACCUM" 1 2 3 4 6 12; do
  if (( candidate_ga > 0 && TARGET_EFFECTIVE_BATCH % (gpu_count * candidate_ga) == 0 )); then
    candidate_pd=$((TARGET_EFFECTIVE_BATCH / (gpu_count * candidate_ga)))
    if (( candidate_pd > 0 && (gpu_count * candidate_pd) % NUM_GENERATIONS == 0 )); then
      PER_DEVICE_TRAIN_BATCH_SIZE=$candidate_pd
      GRADIENT_ACCUMULATION_STEPS=$candidate_ga
      break
    fi
  fi
done
if [[ -z "$PER_DEVICE_TRAIN_BATCH_SIZE" ]]; then
  echo "Cannot preserve effective batch=$TARGET_EFFECTIVE_BATCH with $gpu_count GPUs and num_generations=$NUM_GENERATIONS." >&2
  echo "Use 1, 2, 3, 4, or 6 GPUs, or explicitly choose a compatible TARGET_EFFECTIVE_BATCH." >&2
  exit 1
fi
echo "Batch equivalence: GPUs=$gpu_count per_device=$PER_DEVICE_TRAIN_BATCH_SIZE gradient_accumulation=$GRADIENT_ACCUMULATION_STEPS effective_global_batch=$TARGET_EFFECTIVE_BATCH"

jobs=(
  "swift|math|4000|spg/slurm_scripts/swift/clean_run_math_swift_num_t3_semi0.85_mask_answer_low_confidence_early_0.95.sh"
  "swift|gsm8k|6000|spg/slurm_scripts/swift/clean_run_gsm8k_swift_num_t3_semi0.95_mask_answer_low_confidence_early_0.95.sh"
  "swift|countdown|6000|spg/slurm_scripts/swift/clean_run_countdown_swift_num_t3_semi0.95_mask_answer_low_confidence_early_0.95.sh"
  "diffu_grpo|math|4000|spg/slurm_scripts/new_d1/clean_run_math_d1_num_t3.sh"
  "diffu_grpo|gsm8k|6000|spg/slurm_scripts/new_d1/clean_run_gsm8k_d1_num_t3.sh"
  "diffu_grpo|countdown|6000|spg/slurm_scripts/new_d1/clean_run_countdown_d1_num_t3.sh"
  "token_spg|math|4000|spg/slurm_scripts/token_spg_mix/run_math_base_token_spg_mix_beta1.5_weight0.5_num_t3.sh"
  "token_spg|gsm8k|6000|spg/slurm_scripts/token_spg_mix/run_gsm8k_base_token_spg_mix_beta1.5_weight0.5_num_t3.sh"
  "token_spg|countdown|6000|spg/slurm_scripts/token_spg_mix/run_countdown_base_token_spg_mix_beta1.5_weight0.5_num_t3.sh"
  "elbo|math|4000|spg/slurm_scripts/elbo/clean_run_math_elbo_num_t3.sh"
  "elbo|gsm8k|6000|spg/slurm_scripts/elbo/clean_run_gsm8k_elbo_num_t3.sh"
  "elbo|countdown|6000|spg/slurm_scripts/elbo/clean_run_countdown_elbo_num_t3.sh"
)

echo "Selected GPUs $GPU_ID are free enough; jobs will run serially."
for item in "${jobs[@]}"; do
  IFS='|' read -r trainer dataset max_steps launcher <<<"$item"
  [[ -x "$ROOT/$launcher" ]] || chmod +x "$ROOT/$launcher"
  echo "=== $trainer / $dataset / max_steps=$max_steps ==="
  cmd=(env GPU_IDS="$GPU_ID" CONDA_ENV="$CONDA_ENV" MODEL_PATH="$MODEL_PATH"
       NUM_GENERATIONS="$NUM_GENERATIONS"
       PER_DEVICE_TRAIN_BATCH_SIZE="$PER_DEVICE_TRAIN_BATCH_SIZE"
       GRADIENT_ACCUMULATION_STEPS="$GRADIENT_ACCUMULATION_STEPS"
       LOGITS_MICRO_BATCH_SIZE="${LOGITS_MICRO_BATCH_SIZE:-1}"
       "$ROOT/$launcher")
  printf '%q ' "${cmd[@]}"; echo
  if [[ "$RUN" == 1 ]]; then
    "${cmd[@]}"
    used_mb="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' -v id="$GPU_ID" '$1 == id {print $2}')"
    (( used_mb <= ${GPU_FREE_THRESHOLD_MB:-1024} )) || { echo "GPU became occupied; stopping." >&2; exit 2; }
  fi
done
