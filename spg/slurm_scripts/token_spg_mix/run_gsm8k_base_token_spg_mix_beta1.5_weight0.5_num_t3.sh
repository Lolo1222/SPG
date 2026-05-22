#!/usr/bin/env bash
set -euo pipefail

# Simple runner for token-level SPG (mix objective).
# - Removes SLURM/srun and writes logs to a timestamped file under ../../logs
# - Activates conda env (default: spg)
# - Generates a random main_process_port

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

LOGDIR="${REPO_ROOT}/logs"
mkdir -p "$LOGDIR"
echo "LOGDIR=$LOGDIR"
echo "REPO_ROOT=$REPO_ROOT"

############# FIX *****************
ACCEL_CONFIG_FILE="${REPO_ROOT}/spg/slurm_scripts/accelerate_genai_a100.yaml"
TRAIN_CONFIG_FILE="${REPO_ROOT}/spg/slurm_scripts/train_elbo.yaml"
ENTRYPOINT_PY="${REPO_ROOT}/spg/slurm_scripts/accelerate_entrypoint.py"

DATASET="gsm8k"
RUN_NAME="${DATASET}_base_token_spg_mix_num_t3_beta1.5_weight0.5"
TRAINER="token_spg"
FORWARD_TYPE="random"
MAX_STEPS=6000

NUM_T=3
MIN_T=0
MAX_T=1
NUM_GENERATIONS=6

LOGP_ESTIMATION="mix"
MIX_WEIGHT=0.5
EUBO_BETA=1.5
BETA=0.0
SEMI_OFFLINE_FLAG="False"
SEMI_OFFLINE_DATA_PATH="dataset/llada_gsm8k_generations_7500_converted.jsonl"
SEMI_OFFLINE_RATIO=0.95
SEMI_OFFLINE_STRATEGY="mask_answer_low_confidence"
EARLY_STOP_ROLLOUT_FLAG="False"
EARLY_STOP_THRESHOLD=0.95

MODEL_PATH="${REPO_ROOT}/save_dir/hf_models/LLaDA-8B-Instruct"
CONDA_ENV_DEFAULT="spg"
DEFAULT_NUM_ITER=4
DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE=6
DEFAULT_GRADIENT_ACCUMULATION_STEPS=2
############# FIX *****************

GPU_IDS="${GPU_IDS:-}"
CONDA_ENV="${CONDA_ENV:-$CONDA_ENV_DEFAULT}"
DRY_RUN="${DRY_RUN:-0}"

if [ -n "$GPU_IDS" ]; then
	export CUDA_VISIBLE_DEVICES="$GPU_IDS"
	echo "Using GPUs: $GPU_IDS (CUDA_VISIBLE_DEVICES set)"
else
	echo "Using all visible GPUs (CUDA_VISIBLE_DEVICES not set)"
fi

if command -v conda >/dev/null 2>&1; then
	CONDA_BASE=$(conda info --base 2>/dev/null || true)
	if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
		# shellcheck source=/dev/null
		source "$CONDA_BASE/etc/profile.d/conda.sh"
	fi

	if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
		echo "Activating conda env: ${CONDA_ENV}"
		if ! conda activate "${CONDA_ENV}" 2>/dev/null; then
			echo "Warning: 'conda activate ${CONDA_ENV}' failed; will use 'conda run'"
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
	echo "ERROR: conda not found in PATH." >&2
	exit 1
fi

if [ "${CONDA_ACTIVATED:-0}" != "1" ]; then
	USE_CONDA_RUN=1
else
	USE_CONDA_RUN=0
fi

echo "CONDA_ENV=${CONDA_ENV} (activated=${CONDA_ACTIVATED:-0})"
echo "Python: $(command -v python || true)"

RANDOM_PORT=$((RANDOM % 55536 + 10000))
echo "Using random main_process_port: $RANDOM_PORT"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

SAVE_DIR_BASE="${REPO_ROOT}/save_dir"
SAVE_DIR="${SAVE_DIR_BASE}/token_spg_mix_num_t3_${TIMESTAMP}"
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
export CHECKPOINT_DIR="$SAVE_DIR"

NUM_ITER="${NUM_ITER:-$DEFAULT_NUM_ITER}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-$DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-$DEFAULT_GRADIENT_ACCUMULATION_STEPS}"

LOGFILE="${LOGDIR}/token_spg_mix_num_t3_${TIMESTAMP}.out"
echo "Logging to $LOGFILE"

if [ ! -f "$ENTRYPOINT_PY" ]; then
	echo "ERROR: expected entrypoint script $ENTRYPOINT_PY not found" >&2
	exit 1
fi

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/spg:${PYTHONPATH:-}"

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
	--num_t "$NUM_T"
	--min_t "$MIN_T"
	--max_t "$MAX_T"
	--num_generations "$NUM_GENERATIONS"
	--per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE}
	--gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS}
	--beta "$BETA"
	--logp_estimation "$LOGP_ESTIMATION"
	--mix_weight "$MIX_WEIGHT"
	--eubo_beta "$EUBO_BETA"
	--semi_offline_flag "$SEMI_OFFLINE_FLAG"
	--semi_offline_data_path "$SEMI_OFFLINE_DATA_PATH"
	--semi_offline_ratio "$SEMI_OFFLINE_RATIO"
	--semi_offline_strategy "$SEMI_OFFLINE_STRATEGY"
	--max_steps "$MAX_STEPS"
	--early_stop_rollout_flag "$EARLY_STOP_ROLLOUT_FLAG"
	--early_stop_threshold "$EARLY_STOP_THRESHOLD")

echo
echo "Final command to run:"
printf '%q ' "${ACCEL_CMD[@]}"
echo

if [ "${USE_CONDA_RUN:-0}" = "1" ]; then
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
