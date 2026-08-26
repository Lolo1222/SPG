#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# shellcheck source=../stable_training_lib.sh
source "${REPO_ROOT}/spg/slurm_scripts/stable_training_lib.sh"

ACCEL_CONFIG_FILE="${REPO_ROOT}/spg/slurm_scripts/accelerate_genai_a100.yaml"
TRAIN_CONFIG_FILE="${REPO_ROOT}/spg/slurm_scripts/train_elbo.yaml"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/save_dir/hf_models/LLaDA-8B-Instruct}"
DATASET="math"
RUN_NAME="${RUN_NAME:-${DATASET}_diffu_grpo}"
TRAINER="diffu_grpo"
FORWARD_TYPE="random"

GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-}}"
CONDA_ENV="${CONDA_ENV:-spg}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE_TEST="${SMOKE_TEST:-0}"
NUM_ITER="${NUM_ITER:-4}"

if [ "$SMOKE_TEST" = "1" ]; then
  NUM_ITER="${NUM_ITER_OVERRIDE:-1}"
  NUM_GENERATIONS="${NUM_GENERATIONS:-2}"
  MAX_STEPS="${MAX_STEPS:-1}"
  MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-2}"
  DIFFUSION_STEPS="${DIFFUSION_STEPS:-2}"
  MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-32}"
  BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
  GPU_MEMORY_RESERVE_FRACTION="${GPU_MEMORY_RESERVE_FRACTION:-0.02}"
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/spg:${PYTHONPATH:-}"
spg_activate_conda "$CONDA_ENV"
spg_detect_num_processes
spg_configure_stable_runtime

if [ ! -d "$MODEL_PATH" ]; then
  echo "ERROR: model directory not found: $MODEL_PATH" >&2
  exit 1
fi

LOGDIR="${REPO_ROOT}/logs"
mkdir -p "$LOGDIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="${LOGDIR}/${RUN_NAME}_${TIMESTAMP}.out"
SAVE_DIR="${REPO_ROOT}/save_dir/run_results/${RUN_NAME}_${TIMESTAMP}"
if [ -n "${RESUME_DIR:-}" ]; then
  if [ ! -d "$RESUME_DIR" ]; then
    echo "ERROR: RESUME_DIR does not exist: $RESUME_DIR" >&2
    exit 1
  fi
  SAVE_DIR="$RESUME_DIR"
fi
mkdir -p "$SAVE_DIR"
export CHECKPOINT_DIR="$SAVE_DIR"

ENTRYPOINT_PY="${REPO_ROOT}/spg/slurm_scripts/accelerate_entrypoint.py"
ACCEL_CMD=(accelerate launch
  --config_file "$ACCEL_CONFIG_FILE"
  --main_process_port "$((RANDOM % 55536 + 10000))"
  --num_processes "$NUM_PROCESSES"
  "$ENTRYPOINT_PY"
  --config "$TRAIN_CONFIG_FILE"
  --model_path "$MODEL_PATH"
  --num_iterations "$NUM_ITER"
  --dataset "$DATASET"
  --run_name "$RUN_NAME"
  --output_dir "$SAVE_DIR"
  --trainer "$TRAINER"
  --forward_type "$FORWARD_TYPE"
  --num_generations "$NUM_GENERATIONS"
  --generation_batch_size "$GENERATION_BATCH_SIZE"
  --logits_micro_batch_size "$LOGITS_MICRO_BATCH_SIZE"
  --gpu_memory_reserve_fraction "$GPU_MEMORY_RESERVE_FRACTION"
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS")

if [ "$SMOKE_TEST" = "1" ]; then
  ACCEL_CMD+=(
    --max_steps "$MAX_STEPS"
    --max_train_samples "$MAX_TRAIN_SAMPLES"
    --diffusion_steps "$DIFFUSION_STEPS"
    --max_completion_length "$MAX_COMPLETION_LENGTH"
    --block_length "$BLOCK_LENGTH"
    --save_strategy no)
else
  ACCEL_CMD+=(--save_strategy steps --save_steps "$SAVE_EVERY_STEPS")
fi

echo "GPU_IDS=$GPU_IDS"
echo "SAVE_DIR=$SAVE_DIR"
echo "LOGFILE=$LOGFILE"
printf 'Final command: '; printf '%q ' "${ACCEL_CMD[@]}"; echo
spg_run_with_oom_supervisor "$LOGFILE" "$SAVE_DIR" "${ACCEL_CMD[@]}"
