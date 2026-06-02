#!/usr/bin/env bash
# Launch TensorBoard for Qwen2-Audio training runs.
# Run from the project root:  bash qwen/launch_tb.sh [port]

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="${PROJECT_ROOT}/outputs/qwen_musiccaps_finetuned/runs"
PORT="${1:-6006}"

if [ ! -d "$LOGDIR" ]; then
    echo "[ERROR] Log directory not found: $LOGDIR"
    echo "        Start training first: python qwen/train_lora_qwen.py"
    exit 1
fi

echo "=============================================="
echo "  TensorBoard"
echo "  Logdir : $LOGDIR"
echo "  URL    : http://localhost:${PORT}"
echo "  Remote : http://$(hostname -I | awk '{print $1}'):${PORT}"
echo "  Stop   : Ctrl-C"
echo "=============================================="

# Activate venv if present
if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
fi

tensorboard \
    --logdir "$LOGDIR" \
    --port "$PORT" \
    --bind_all \
    --reload_interval 10
