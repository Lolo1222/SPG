#!/usr/bin/env bash
set -euo pipefail

# 简单包装器：设置 CUDA_VISIBLE_DEVICES 并调用 Python 分配脚本
# 用法：GPU_IDS=0,1 ./scripts/debug.sh --mem 2048 --duration 3600

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY_SCRIPT="${REPO_ROOT}/scripts/debug.py"

if [ ! -f "$PY_SCRIPT" ]; then
  echo "错误：未找到 $PY_SCRIPT" >&2
  exit 1
fi

GPU_IDS="${GPU_IDS:-}"
if [ -n "$GPU_IDS" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU_IDS"
  echo "CUDA_VISIBLE_DEVICES=$GPU_IDS"
fi

python3 "$PY_SCRIPT" "$@"
