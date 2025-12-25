#!/usr/bin/env bash
# Wait until a target GPU has at least the requested free memory (in GiB)
# Usage:
#   ./wait_for_gpu_memory.sh <GPU_ID> <FREE_GIB> [POLL_SECONDS]
# Example:
#   ./wait_for_gpu_memory.sh 0 60 10

set -euo pipefail

GPU_ID=${1:-0}
REQUIRED_FREE_GIB=${2:-55}
POLL_SECONDS=${3:-5}

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found" >&2
  exit 2
fi

echo "Waiting for GPU ${GPU_ID} to have >= ${REQUIRED_FREE_GIB} GiB free..."

while true; do
  # Query total, used memory in MiB for the specific GPU
  line=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i "$GPU_ID")
  total_mib=$(echo "$line" | awk -F", " '{print $1}')
  used_mib=$(echo "$line"  | awk -F", " '{print $2}')
  free_mib=$((total_mib - used_mib))
  free_gib=$(awk -v mib="$free_mib" 'BEGIN{printf "%.1f", mib/1024.0}')

  echo "Current free: ${free_gib} GiB (poll ${POLL_SECONDS}s)"

  # Compare using integer MiB to avoid float issues
  required_mib=$((REQUIRED_FREE_GIB * 1024))
  if [ "$free_mib" -ge "$required_mib" ]; then
    echo "Sufficient free memory available on GPU ${GPU_ID}. Proceeding."
    exit 0
  fi

  sleep "$POLL_SECONDS"
done
