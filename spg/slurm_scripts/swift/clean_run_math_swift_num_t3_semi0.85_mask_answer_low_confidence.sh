#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LOGDIR="${REPO_ROOT}/logs"
mkdir -p "$LOGDIR"
echo "LOGDIR=$LOGDIR"
echo "REPO_ROOT=$REPO_ROOT"

############# FIX *****************
ACCEL_CONFIG_FILE="${REPO_ROOT}/spg/slurm_scripts/accelerate_genai_a100.yaml"
TRAIN_CONFIG_FILE="${REPO_ROOT}/spg/slurm_scripts/train_elbo.yaml"
DATASET="math"
RUN_NAME=${DATASET}_swift_grpo_generated_num_t3_semi0.85_mask_answer_low_confidence
TRAINER="swift"
FORWARD_TYPE="random"
SEMI_OFFLINE_STRATEGY="mask_answer_low_confidence"
############# FIX *****************

GPU_IDS="${GPU_IDS:-}"

CONDA_ENV="${CONDA_ENV:-spg}"
# Dry-run mode: set DRY_RUN=1 to only print the command without executing
DRY_RUN="${DRY_RUN:-0}"
if [ -n "$GPU_IDS" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU_IDS"
  echo "Using GPUs: $GPU_IDS (CUDA_VISIBLE_DEVICES set)"
else
  echo "Using all visible GPUs (CUDA_VISIBLE_DEVICES not set)"
fi

echo "Python: $(command -v python || true)"

# random port in same range as sbatch
RANDOM_PORT=$((RANDOM % 55536 + 10000))
echo "Using random main_process_port: $RANDOM_PORT"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)




# timestamped logfile (use same timestamp as SAVE_DIR)
LOGFILE="${LOGDIR}/${RUN_NAME}_${TIMESTAMP}.out"
echo "Logging to $LOGFILE"

# Save dir: under repo save_dir, create a per-run folder spg_mix_<timestamp>
SAVE_DIR_BASE="${REPO_ROOT}/save_dir"
SAVE_DIR="${SAVE_DIR_BASE}/run_results/${RUN_NAME}_${TIMESTAMP}"

# If RESUME_DIR is set, use it as SAVE_DIR (resume run) and ensure it exists
if [ -n "${RESUME_DIR:-}" ]; then
  if [ -d "${RESUME_DIR}" ]; then
    echo "RESUME_DIR set, resuming from: ${RESUME_DIR}"
    SAVE_DIR="${RESUME_DIR}"
  else
    echo "ERROR: RESUME_DIR=${RESUME_DIR} does not exist" >&2
    exit 1
  fi
fi

mkdir -p "$SAVE_DIR"

# Export CHECKPOINT_DIR so the training script will search this directory for checkpoints
export CHECKPOINT_DIR="$SAVE_DIR"


MODEL_PATH="/home/jwliu/dlm/SPG/save_dir/hf_models/LLaDA-8B-Instruct"
# Allow overriding for quick smoke-tests
NUM_ITER="${NUM_ITER:-4}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-6}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"



# Ensure repo root and the spg package dir are on PYTHONPATH so imports work
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/spg:${PYTHONPATH:-}"

# Use absolute python executable to avoid accelerate/torch treating 'python' as a relative script
PYTHON_EXEC="$(command -v python || true)"
if [ -z "$PYTHON_EXEC" ]; then
  PYTHON_EXEC="python"
fi

# Determine how many processes to launch with accelerate.
# If the user specified GPU_IDS (e.g. "1" or "0,1"), use its count. Otherwise
# fall back to counting visible GPUs via nvidia-smi. Default to 1 if detection fails.
if [ -n "${GPU_IDS:-}" ]; then
  NUM_PROCESSES=$(echo "$GPU_IDS" | awk -F',' '{print NF}')
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_PROCESSES=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
  else
    NUM_PROCESSES=1
  fi
fi
if [ -z "$NUM_PROCESSES" ] || [ "$NUM_PROCESSES" -lt 1 ]; then
  NUM_PROCESSES=1
fi
echo "Launching accelerate with num_processes=$NUM_PROCESSES"

# Build the accelerate command array (we may prefix it with conda run later).
# Use a small Python wrapper entrypoint so the launcher executes a normal script
# (avoids trying to open the interpreter binary as a script) and that wrapper
# will run the module as a module so relative imports work.
ENTRYPOINT_PY="${REPO_ROOT}/spg/slurm_scripts/accelerate_entrypoint.py"
if [ ! -f "$ENTRYPOINT_PY" ]; then
  echo "ERROR: expected entrypoint script $ENTRYPOINT_PY not found" >&2
  exit 1
fi

ACCEL_BASE=(accelerate launch
  --config_file "$ACCEL_CONFIG_FILE"
  --main_process_port "$RANDOM_PORT"
  --num_processes "$NUM_PROCESSES"
  "$ENTRYPOINT_PY")

ACCEL_CMD=("${ACCEL_BASE[@]}"
  --config "$TRAIN_CONFIG_FILE"
  --model_path "$MODEL_PATH"
  --num_iterations "$NUM_ITER"
  --dataset "$DATASET"
  --run_name "$RUN_NAME"
  --output_dir "${SAVE_DIR}"
  --trainer "$TRAINER"
  --forward_type "$FORWARD_TYPE"
  --num_t 3
  --min_t 0
  --max_t 1
  --num_generations 6
  --semi_offline_flag True
  --semi_offline_data_path dataset/llada_math_generations_7500_converted.jsonl
  --semi_offline_ratio 0.85
  --semi_offline_strategy ${SEMI_OFFLINE_STRATEGY}
  --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE}
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS})

echo
echo "Final command to run:"
printf '%q ' "${ACCEL_CMD[@]}"
echo
CMD=("${ACCEL_CMD[@]}")
if [ "$DRY_RUN" != "1" ]; then
  "${CMD[@]}" 2>&1 | tee "$LOGFILE"
else
  echo "DRY_RUN=1: not executing accelerate; printed command only. Logs will not be written."
fi

# End of script
