#!/usr/bin/env bash

# Shared runtime helpers for long-running local Accelerate jobs. Source this
# file after REPO_ROOT, GPU_IDS and the experiment-specific variables are set.

spg_activate_conda() {
  local env_name=$1
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found in PATH" >&2
    return 1
  fi

  local conda_base
  conda_base=$(conda info --base 2>/dev/null || true)
  if [ -n "$conda_base" ] && [ -f "$conda_base/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "$conda_base/etc/profile.d/conda.sh"
  fi
  if ! conda env list | awk '{print $1}' | grep -qx "$env_name"; then
    echo "ERROR: conda environment '$env_name' not found" >&2
    conda env list >&2 || true
    return 1
  fi
  conda activate "$env_name"
  echo "Using conda environment: $env_name ($(command -v python))"
}

spg_detect_num_processes() {
  if [ -z "${GPU_IDS:-}" ]; then
    echo "ERROR: GPU_IDS (or CUDA_VISIBLE_DEVICES) must explicitly select the GPU(s) on a shared machine" >&2
    return 1
  fi
  NUM_PROCESSES=$(awk -F',' '{print NF}' <<< "$GPU_IDS")
  if ! [[ "$NUM_PROCESSES" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: unable to determine process count from GPU_IDS='$GPU_IDS'" >&2
    return 1
  fi
}

spg_gcd() {
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

spg_configure_stable_runtime() {
  local per_device_was_set=0
  local grad_accum_was_set=0
  if [ -n "${PER_DEVICE_TRAIN_BATCH_SIZE+x}" ]; then
    per_device_was_set=1
  fi
  if [ -n "${GRADIENT_ACCUMULATION_STEPS+x}" ]; then
    grad_accum_was_set=1
  fi

  NUM_GENERATIONS="${NUM_GENERATIONS:-6}"
  GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-1}"
  LOGITS_MICRO_BATCH_SIZE="${LOGITS_MICRO_BATCH_SIZE:-1}"
  GPU_MEMORY_RESERVE_FRACTION="${GPU_MEMORY_RESERVE_FRACTION:-0.90}"
  SEMI_OFFLINE_MASKING_BATCH_SIZE="${SEMI_OFFLINE_MASKING_BATCH_SIZE:-1}"
  SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-100}"
  AUTO_RESTART_ON_OOM="${AUTO_RESTART_ON_OOM:-1}"
  MAX_OOM_RESTARTS="${MAX_OOM_RESTARTS:-20}"
  OOM_RETRY_SECONDS="${OOM_RETRY_SECONDS:-30}"
  BASELINE_MEMORY_MARGIN_MIB="${BASELINE_MEMORY_MARGIN_MIB:-1024}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

  local setting value
  for setting in NUM_GENERATIONS GENERATION_BATCH_SIZE LOGITS_MICRO_BATCH_SIZE SEMI_OFFLINE_MASKING_BATCH_SIZE SAVE_EVERY_STEPS MAX_OOM_RESTARTS OOM_RETRY_SECONDS BASELINE_MEMORY_MARGIN_MIB; do
    value=${!setting}
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
      echo "ERROR: $setting must be a non-negative integer, got '$value'" >&2
      return 1
    fi
  done
  if [ "$NUM_GENERATIONS" -lt 2 ] || [ "$GENERATION_BATCH_SIZE" -lt 1 ] || [ "$LOGITS_MICRO_BATCH_SIZE" -lt 1 ] || [ "$SEMI_OFFLINE_MASKING_BATCH_SIZE" -lt 1 ] || [ "$SAVE_EVERY_STEPS" -lt 1 ]; then
    echo "ERROR: generation count must be >=2 and batch/save controls must be >=1" >&2
    return 1
  fi
  if ! [[ "$GPU_MEMORY_RESERVE_FRACTION" =~ ^([0-9]+([.][0-9]+)?|[.][0-9]+)$ ]] || ! awk -v value="$GPU_MEMORY_RESERVE_FRACTION" 'BEGIN {exit !(value >= 0 && value < 1)}'; then
    echo "ERROR: GPU_MEMORY_RESERVE_FRACTION must be in [0, 1), got '$GPU_MEMORY_RESERVE_FRACTION'" >&2
    return 1
  fi

  if [ "$per_device_was_set" -eq 0 ]; then
    local group_process_gcd
    group_process_gcd=$(spg_gcd "$NUM_PROCESSES" "$NUM_GENERATIONS")
    PER_DEVICE_TRAIN_BATCH_SIZE=$((NUM_GENERATIONS / group_process_gcd))
  elif ! [[ "$PER_DEVICE_TRAIN_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: PER_DEVICE_TRAIN_BATCH_SIZE must be a positive integer" >&2
    return 1
  fi

  GLOBAL_GENERATION_BATCH=$((NUM_PROCESSES * PER_DEVICE_TRAIN_BATCH_SIZE))
  if [ $((GLOBAL_GENERATION_BATCH % NUM_GENERATIONS)) -ne 0 ]; then
    echo "ERROR: global batch $GLOBAL_GENERATION_BATCH must be divisible by NUM_GENERATIONS=$NUM_GENERATIONS" >&2
    return 1
  fi

  if [ "$grad_accum_was_set" -eq 0 ]; then
    local target_effective_batch=$((NUM_GENERATIONS * 2))
    if [ "$GLOBAL_GENERATION_BATCH" -lt "$target_effective_batch" ]; then
      GRADIENT_ACCUMULATION_STEPS=$((target_effective_batch / GLOBAL_GENERATION_BATCH))
    else
      GRADIENT_ACCUMULATION_STEPS=1
    fi
  elif ! [[ "$GRADIENT_ACCUMULATION_STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: GRADIENT_ACCUMULATION_STEPS must be a positive integer" >&2
    return 1
  fi

  echo "Batch layout: processes=$NUM_PROCESSES, per_device=$PER_DEVICE_TRAIN_BATCH_SIZE, global=$GLOBAL_GENERATION_BATCH, generations=$NUM_GENERATIONS, grad_accum=$GRADIENT_ACCUMULATION_STEPS"
  echo "VRAM controls: generation_batch=$GENERATION_BATCH_SIZE, logits_micro_batch=$LOGITS_MICRO_BATCH_SIZE, reserve_fraction=$GPU_MEMORY_RESERVE_FRACTION, allocator=$PYTORCH_CUDA_ALLOC_CONF"
}

spg_wait_for_baseline_memory() {
  local gpu_id current_free required_free all_ready
  while true; do
    all_ready=1
    for gpu_id in "${SPG_MONITORED_GPU_IDS[@]}"; do
      if [ -z "${SPG_BASELINE_FREE_MIB[$gpu_id]+x}" ]; then
        continue
      fi
      required_free=$((SPG_BASELINE_FREE_MIB[$gpu_id] - BASELINE_MEMORY_MARGIN_MIB))
      if [ "$required_free" -lt 0 ]; then
        required_free=0
      fi
      current_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | head -n 1 | tr -d ' ' || true)
      if ! [[ "$current_free" =~ ^[0-9]+$ ]] || [ "$current_free" -lt "$required_free" ]; then
        echo "GPU $gpu_id has ${current_free:-unknown} MiB free; waiting for retry baseline ${required_free} MiB"
        all_ready=0
      fi
    done
    if [ "$all_ready" -eq 1 ]; then
      return 0
    fi
    sleep "$OOM_RETRY_SECONDS"
  done
}

spg_run_with_oom_supervisor() {
  local logfile=$1
  local save_dir=$2
  shift 2
  local -a command=("$@")

  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN=1: command validated but not executed"
    return 0
  fi

  IFS=',' read -r -a SPG_MONITORED_GPU_IDS <<< "$GPU_IDS"
  declare -gA SPG_BASELINE_FREE_MIB=()
  local gpu_id free_mib
  for gpu_id in "${SPG_MONITORED_GPU_IDS[@]}"; do
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | head -n 1 | tr -d ' ' || true)
    if [[ "$free_mib" =~ ^[0-9]+$ ]]; then
      SPG_BASELINE_FREE_MIB["$gpu_id"]=$free_mib
      echo "GPU $gpu_id retry baseline: ${free_mib} MiB free"
    fi
  done

  local oom_restarts=0
  local run_attempt=1
  local attempt_log train_status
  while true; do
    attempt_log="${logfile}.attempt_${run_attempt}"
    echo "[supervisor] training attempt $run_attempt" | tee -a "$logfile"
    set +e
    "${command[@]}" 2>&1 | tee -a "$logfile" "$attempt_log"
    train_status=${PIPESTATUS[0]}
    set -e

    if [ "$train_status" -eq 0 ]; then
      echo "[supervisor] training completed successfully" | tee -a "$logfile"
      return 0
    fi
    if ! grep -Eqi 'CUDA out of memory|torch[.]OutOfMemoryError|CUDNN_STATUS_ALLOC_FAILED|CUDA error: out of memory' "$attempt_log"; then
      echo "[supervisor] non-OOM failure with status $train_status; not restarting" | tee -a "$logfile" >&2
      return "$train_status"
    fi
    if [ "$AUTO_RESTART_ON_OOM" != "1" ] || [ "$oom_restarts" -ge "$MAX_OOM_RESTARTS" ]; then
      echo "[supervisor] OOM restart disabled or exhausted ($oom_restarts/$MAX_OOM_RESTARTS)" | tee -a "$logfile" >&2
      return "$train_status"
    fi

    oom_restarts=$((oom_restarts + 1))
    run_attempt=$((run_attempt + 1))
    echo "[supervisor] OOM detected; restart $oom_restarts/$MAX_OOM_RESTARTS will resume from $save_dir" | tee -a "$logfile"
    sleep "$OOM_RETRY_SECONDS"
    spg_wait_for_baseline_memory
  done
}
