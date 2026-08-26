#!/usr/bin/env bash
set -euo pipefail

# Simple runner equivalent to the sbatch script math_base_spg_eubo_beta1.5.sbatch
# - Removes SLURM/srun and writes logs to a timestamped file under ../../logs
# - Attempts to activate conda env `spg` (works with modern `conda activate` and older `source activate`)
# - Generates a random main_process_port like the original

# Resolve repository root (script is in spg/slurm_scripts/spg_eubo)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Use repo-root-based paths so this script can be invoked from the repo root
LOGDIR="${REPO_ROOT}/logs"
mkdir -p "$LOGDIR"
echo "LOGDIR=$LOGDIR"
echo "REPO_ROOT=$REPO_ROOT"

# User-configurable: set GPU_IDS here (e.g. "0" or "0,1") or export GPU_IDS in the environment.
# Honor an existing CUDA_VISIBLE_DEVICES selection when GPU_IDS is omitted.
# Example: GPU_IDS=0 ./run_math_base_spg_eubo_beta1.5.sh
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-}}"
# Which conda env to use (default: spg). Can override with CONDA_ENV=myenv
CONDA_ENV="${CONDA_ENV:-spg}"
# Dry-run mode: set DRY_RUN=1 to only print the command without executing
DRY_RUN="${DRY_RUN:-0}"
# Favor allocator segments that can grow without leaving many unusable slivers.
# Respect an explicitly supplied allocator configuration.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [ -n "$GPU_IDS" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU_IDS"
  echo "Using GPUs: $GPU_IDS (CUDA_VISIBLE_DEVICES set)"
else
  echo "Using all visible GPUs (CUDA_VISIBLE_DEVICES not set)"
fi

# Try to activate conda env `spg` (compatible with conda >=4.4)
if command -v conda >/dev/null 2>&1; then
  # load conda functions into shell
  CONDA_BASE=$(conda info --base 2>/dev/null || true)
  if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    # shell hook
    # shellcheck source=/dev/null
    source "$CONDA_BASE/etc/profile.d/conda.sh"
  fi

  # Verify that the requested conda env exists
  if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    echo "Activating conda env: ${CONDA_ENV}"
    # Try standard activate; if it fails, fall back to conda run below
    if ! conda activate "${CONDA_ENV}" 2>/dev/null; then
      echo "Warning: 'conda activate ${CONDA_ENV}' failed; will attempt 'conda run' when launching accelerate"
      CONDA_ACTIVATED=0
    else
      CONDA_ACTIVATED=1
    fi
  else
    echo "ERROR: conda environment '${CONDA_ENV}' not found. Available envs:" >&2
    conda env list || true
    exit 1
  fi
else
  echo "ERROR: conda not found in PATH. Please install conda or ensure it's available." >&2
  exit 1
fi

if [ "${CONDA_ACTIVATED:-0}" != "1" ]; then
  # We'll use 'conda run -n <env> --no-capture-output' to execute accelerate if activation failed
  USE_CONDA_RUN=1
else
  USE_CONDA_RUN=0
fi

echo "CONDA_ENV=${CONDA_ENV}  (activated=${CONDA_ACTIVATED:-0})"
echo "Python: $(command -v python || true)"

# Print some diagnostics
echo "Conda env: $(conda info --envs 2>/dev/null | grep '*' || true)"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total --format=csv || true

# random port in same range as sbatch
RANDOM_PORT=$((RANDOM % 55536 + 10000))
echo "Using random main_process_port: $RANDOM_PORT"

# timestamp to use for SAVE_DIR and logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Save dir: under repo save_dir, create a per-run folder spg_eubo_<timestamp>
SAVE_DIR_BASE="${REPO_ROOT}/save_dir"
SAVE_DIR="${SAVE_DIR_BASE}/d1_${TIMESTAMP}"
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

DATASET="math"
RUN_NAME=${DATASET}_base_d1_beta1.5
# Use a shared model path under repo save_dir/hf_models (not inside per-run folder)
MODEL_PATH="${REPO_ROOT}/save_dir/hf_models/LLaDA-8B-Instruct"
# Allow overriding for quick smoke-tests
NUM_ITER="${NUM_ITER:-4}"
NUM_GENERATIONS="${NUM_GENERATIONS:-6}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-1}"
LOGITS_MICRO_BATCH_SIZE="${LOGITS_MICRO_BATCH_SIZE:-1}"
GPU_MEMORY_RESERVE_FRACTION="${GPU_MEMORY_RESERVE_FRACTION:-0}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-20}"
AUTO_RESTART_ON_OOM="${AUTO_RESTART_ON_OOM:-1}"
MAX_OOM_RESTARTS="${MAX_OOM_RESTARTS:-20}"
OOM_RETRY_SECONDS="${OOM_RETRY_SECONDS:-30}"
BASELINE_MEMORY_MARGIN_MIB="${BASELINE_MEMORY_MARGIN_MIB:-1024}"

# Remember which batch controls the caller explicitly set. When unset, they
# are derived below so that a fixed GRPO generation group is distributed over
# all processes instead of being repeated in full on every GPU.
PER_DEVICE_BATCH_WAS_SET=0
if [ -n "${PER_DEVICE_TRAIN_BATCH_SIZE+x}" ]; then
  PER_DEVICE_BATCH_WAS_SET=1
else
  PER_DEVICE_TRAIN_BATCH_SIZE=""
fi
GRAD_ACCUM_WAS_SET=0
if [ -n "${GRADIENT_ACCUMULATION_STEPS+x}" ]; then
  GRAD_ACCUM_WAS_SET=1
else
  GRADIENT_ACCUMULATION_STEPS=""
fi

# timestamped logfile (use same timestamp as SAVE_DIR)
LOGFILE="${LOGDIR}/d1_${TIMESTAMP}.out"
echo "Logging to $LOGFILE"

# Run accelerate directly (no srun / SLURM). Adjust --config_file and script path if needed.
ACCEL_CONFIG_FILE="${REPO_ROOT}/spg/slurm_scripts/accelerate_genai_a100.yaml"
TRAIN_CONFIG_FILE="${REPO_ROOT}/spg/slurm_scripts/train_max_step.yaml"

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

if ! [[ "$NUM_GENERATIONS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: NUM_GENERATIONS must be a positive integer, got '$NUM_GENERATIONS'" >&2
  exit 1
fi
for numeric_setting in GENERATION_BATCH_SIZE LOGITS_MICRO_BATCH_SIZE SAVE_EVERY_STEPS MAX_OOM_RESTARTS OOM_RETRY_SECONDS BASELINE_MEMORY_MARGIN_MIB; do
  numeric_value=${!numeric_setting}
  if ! [[ "$numeric_value" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $numeric_setting must be a non-negative integer, got '$numeric_value'" >&2
    exit 1
  fi
done
if ! [[ "$GPU_MEMORY_RESERVE_FRACTION" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)$ ]] || ! awk -v value="$GPU_MEMORY_RESERVE_FRACTION" 'BEGIN {exit !(value >= 0 && value < 1)}'; then
  echo "ERROR: GPU_MEMORY_RESERVE_FRACTION must be a number in [0, 1), got '$GPU_MEMORY_RESERVE_FRACTION'" >&2
  exit 1
fi
if [ "$GENERATION_BATCH_SIZE" -lt 1 ] || [ "$LOGITS_MICRO_BATCH_SIZE" -lt 1 ] || [ "$SAVE_EVERY_STEPS" -lt 1 ]; then
  echo "ERROR: GENERATION_BATCH_SIZE, LOGITS_MICRO_BATCH_SIZE, and SAVE_EVERY_STEPS must be at least 1" >&2
  exit 1
fi
if [ "$PER_DEVICE_BATCH_WAS_SET" -eq 1 ] && ! [[ "$PER_DEVICE_TRAIN_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: PER_DEVICE_TRAIN_BATCH_SIZE must be a positive integer, got '$PER_DEVICE_TRAIN_BATCH_SIZE'" >&2
  exit 1
fi
if [ "$GRAD_ACCUM_WAS_SET" -eq 1 ] && ! [[ "$GRADIENT_ACCUMULATION_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: GRADIENT_ACCUMULATION_STEPS must be a positive integer, got '$GRADIENT_ACCUMULATION_STEPS'" >&2
  exit 1
fi

gcd() {
  local a=$1
  local b=$2
  local remainder
  while [ "$b" -ne 0 ]; do
    remainder=$((a % b))
    a=$b
    b=$remainder
  done
  echo "$a"
}

if [ "$PER_DEVICE_BATCH_WAS_SET" -eq 0 ]; then
  GROUP_PROCESS_GCD=$(gcd "$NUM_PROCESSES" "$NUM_GENERATIONS")
  PER_DEVICE_TRAIN_BATCH_SIZE=$((NUM_GENERATIONS / GROUP_PROCESS_GCD))
fi
GLOBAL_GENERATION_BATCH=$((NUM_PROCESSES * PER_DEVICE_TRAIN_BATCH_SIZE))
if [ $((GLOBAL_GENERATION_BATCH % NUM_GENERATIONS)) -ne 0 ]; then
  echo "ERROR: num_processes * PER_DEVICE_TRAIN_BATCH_SIZE (${GLOBAL_GENERATION_BATCH}) must be divisible by NUM_GENERATIONS (${NUM_GENERATIONS})." >&2
  exit 1
fi

# Preserve the original effective batch of 12 where possible. For process
# counts such as 4 or 8, the smallest valid global generation batch is already
# 12 or 24, so one accumulation step is the closest valid choice.
if [ "$GRAD_ACCUM_WAS_SET" -eq 0 ]; then
  TARGET_EFFECTIVE_BATCH=$((NUM_GENERATIONS * 2))
  if [ "$GLOBAL_GENERATION_BATCH" -lt "$TARGET_EFFECTIVE_BATCH" ]; then
    GRADIENT_ACCUMULATION_STEPS=$((TARGET_EFFECTIVE_BATCH / GLOBAL_GENERATION_BATCH))
  else
    GRADIENT_ACCUMULATION_STEPS=1
  fi
fi

echo "Launching accelerate with num_processes=$NUM_PROCESSES"
echo "Batch layout: per_device=$PER_DEVICE_TRAIN_BATCH_SIZE, global=$GLOBAL_GENERATION_BATCH, generations=$NUM_GENERATIONS, grad_accum=$GRADIENT_ACCUMULATION_STEPS"
echo "VRAM controls: generation_batch=$GENERATION_BATCH_SIZE, logits_micro_batch=$LOGITS_MICRO_BATCH_SIZE, reserve_fraction=$GPU_MEMORY_RESERVE_FRACTION, allocator=$PYTORCH_CUDA_ALLOC_CONF"

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
  --trainer diffu_grpo
  --forward_type block_random
  --num_generations "$NUM_GENERATIONS"
  --generation_batch_size "$GENERATION_BATCH_SIZE"
  --logits_micro_batch_size "$LOGITS_MICRO_BATCH_SIZE"
  --gpu_memory_reserve_fraction "$GPU_MEMORY_RESERVE_FRACTION"
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --save_strategy steps
  --save_steps "$SAVE_EVERY_STEPS")

echo
echo "Final command to run:"
printf '%q ' "${ACCEL_CMD[@]}"
echo

# If we couldn't actually activate the env, prefix with conda run to ensure the command runs inside the env
if [ "${USE_CONDA_RUN:-0}" = "1" ]; then
  # Note: --no-capture-output preserves stdout/stderr behavior
  CMD=(conda run -n "${CONDA_ENV}" --no-capture-output "${ACCEL_CMD[@]}")
else
  CMD=("${ACCEL_CMD[@]}")
fi

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN=1: not executing accelerate; printed command only. Logs will not be written."
  exit 0
fi

# Record the amount of free memory that made the initial launch possible. If a
# co-tenant later causes an OOM, wait for approximately this baseline to return
# before relaunching. The trainer automatically resumes from SAVE_DIR.
MONITORED_GPU_IDS=()
if command -v nvidia-smi >/dev/null 2>&1; then
  if [ -n "$GPU_IDS" ]; then
    IFS=',' read -r -a MONITORED_GPU_IDS <<< "$GPU_IDS"
  else
    mapfile -t MONITORED_GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
  fi
fi
declare -A BASELINE_FREE_MIB=()
for gpu_id in "${MONITORED_GPU_IDS[@]}"; do
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | head -n 1 | tr -d ' ' || true)
  if [[ "$free_mib" =~ ^[0-9]+$ ]]; then
    BASELINE_FREE_MIB["$gpu_id"]=$free_mib
    echo "GPU $gpu_id retry baseline: ${free_mib} MiB free"
  fi
done

wait_for_baseline_memory() {
  local gpu_id current_free required_free
  while true; do
    local all_ready=1
    for gpu_id in "${MONITORED_GPU_IDS[@]}"; do
      if [ -z "${BASELINE_FREE_MIB[$gpu_id]+x}" ]; then
        continue
      fi
      required_free=$((BASELINE_FREE_MIB[$gpu_id] - BASELINE_MEMORY_MARGIN_MIB))
      if [ "$required_free" -lt 0 ]; then
        required_free=0
      fi
      current_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | head -n 1 | tr -d ' ' || true)
      if ! [[ "$current_free" =~ ^[0-9]+$ ]] || [ "$current_free" -lt "$required_free" ]; then
        echo "GPU $gpu_id has ${current_free:-unknown} MiB free; waiting for ${required_free} MiB retry baseline"
        all_ready=0
      fi
    done
    if [ "$all_ready" -eq 1 ]; then
      return 0
    fi
    sleep "$OOM_RETRY_SECONDS"
  done
}

oom_restarts=0
run_attempt=1
while true; do
  ATTEMPT_LOG="${LOGFILE}.attempt_${run_attempt}"
  echo "[supervisor] training attempt $run_attempt" | tee -a "$LOGFILE"
  set +e
  "${CMD[@]}" 2>&1 | tee -a "$LOGFILE" "$ATTEMPT_LOG"
  train_status=${PIPESTATUS[0]}
  set -e

  if [ "$train_status" -eq 0 ]; then
    echo "[supervisor] training completed successfully" | tee -a "$LOGFILE"
    break
  fi

  if ! grep -Eqi 'CUDA out of memory|torch[.]OutOfMemoryError|CUDNN_STATUS_ALLOC_FAILED|CUDA error: out of memory' "$ATTEMPT_LOG"; then
    echo "[supervisor] training failed with status $train_status for a non-OOM reason; not restarting" | tee -a "$LOGFILE" >&2
    exit "$train_status"
  fi
  if [ "$AUTO_RESTART_ON_OOM" != "1" ] || [ "$oom_restarts" -ge "$MAX_OOM_RESTARTS" ]; then
    echo "[supervisor] OOM detected, but automatic restart is disabled or exhausted ($oom_restarts/$MAX_OOM_RESTARTS)" | tee -a "$LOGFILE" >&2
    exit "$train_status"
  fi

  oom_restarts=$((oom_restarts + 1))
  run_attempt=$((run_attempt + 1))
  echo "[supervisor] OOM detected; restart $oom_restarts/$MAX_OOM_RESTARTS will resume from $SAVE_DIR" | tee -a "$LOGFILE"
  sleep "$OOM_RETRY_SECONDS"
  wait_for_baseline_memory
done

# End of script
