#!/usr/bin/env bash
set -euo pipefail

# Simple runner equivalent to the sbatch script math_base_spg_mix_beta1.5.sbatch
# - Removes SLURM/srun and writes logs to a timestamped file under ../../logs
# - Attempts to activate conda env `spg` (works with modern `conda activate` and older `source activate`)
# - Generates a random main_process_port like the original

# Resolve repository root (script is in spg/slurm_scripts/spg_mix)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Use repo-root-based paths so this script can be invoked from the repo root
LOGDIR="${REPO_ROOT}/logs"
mkdir -p "$LOGDIR"
echo "LOGDIR=$LOGDIR"
echo "REPO_ROOT=$REPO_ROOT"

# User-configurable: set GPU_IDS here (e.g. "0" or "0,1") or export GPU_IDS in the environment
# Example: GPU_IDS=0 ./run_math_base_spg_mix_beta1.5.sh
GPU_IDS="${GPU_IDS:-}"
# Which conda env to use (default: spg). Can override with CONDA_ENV=myenv
CONDA_ENV="${CONDA_ENV:-spg}"
# Dry-run mode: set DRY_RUN=1 to only print the command without executing
DRY_RUN="${DRY_RUN:-0}"
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

# Save dir: under repo save_dir, create a per-run folder spg_mix_<timestamp>
SAVE_DIR_BASE="${REPO_ROOT}/save_dir"
SAVE_DIR="${SAVE_DIR_BASE}/spg_mix_${TIMESTAMP}"
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
RUN_NAME=${DATASET}_base_spg_mix_beta1.5
# Use a shared model path under repo save_dir/hf_models (not inside per-run folder)
MODEL_PATH="${REPO_ROOT}/save_dir/hf_models/LLaDA-8B-Instruct"
# Allow overriding for quick smoke-tests
NUM_ITER="${NUM_ITER:-4}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-6}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"

# timestamped logfile (use same timestamp as SAVE_DIR)
LOGFILE="${LOGDIR}/spg_mix_${TIMESTAMP}.out"
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
  --trainer spg
  --forward_type block_random
  --num_t 2
  --min_t 0
  --max_t 1
  --num_generations 6
  --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE}
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS}
  --beta 0.0
  --logp_estimation mix
  --mix_weight 0.5
  --eubo_beta 1.5)

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

if [ "$DRY_RUN" != "1" ]; then
  "${CMD[@]}" 2>&1 | tee "$LOGFILE"
else
  echo "DRY_RUN=1: not executing accelerate; printed command only. Logs will not be written."
fi

# End of script
