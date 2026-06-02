#!/usr/bin/env bash
# Launch TensorBoard for all Qwen2-Audio training runs.
#
# Usage:
#   bash qwen/launch_tb.sh            # default port 6006
#   bash qwen/launch_tb.sh 6007       # custom port
#   bash qwen/launch_tb.sh 6006 exp-005  # show only one experiment
#
# TensorBoard run naming:
#   Each experiment is saved under runs/EXP_ID/ (e.g. runs/exp-005/).
#   Set EXP_ID at the top of train_lora_qwen.py before training.
#   Old unnamed runs (timestamp-based) remain visible for comparison.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:-6006}"
FILTER="${2:-}"   # optional: show only runs matching this string

# Collect all run directories (named + timestamp-based)
RUNS_DIR="${PROJECT_ROOT}/outputs/qwen_musiccaps_finetuned/runs"

if [ ! -d "$RUNS_DIR" ]; then
    echo "[ERROR] No runs found at: $RUNS_DIR"
    echo "        Train first: python qwen/train_lora_qwen.py"
    exit 1
fi

# Build logdir argument: can be a single dir or "name1:path1,name2:path2"
if [ -n "$FILTER" ]; then
    # Filter to runs matching the given string
    LOGDIR=""
    for d in "$RUNS_DIR"/*/; do
        name="$(basename "$d")"
        if [[ "$name" == *"$FILTER"* ]]; then
            LOGDIR="${LOGDIR}${name}:${d},"
        fi
    done
    LOGDIR="${LOGDIR%,}"   # strip trailing comma
    if [ -z "$LOGDIR" ]; then
        echo "[ERROR] No runs matching '$FILTER'. Available:"
        ls "$RUNS_DIR"
        exit 1
    fi
else
    LOGDIR="$RUNS_DIR"
fi

# Activate venv if present
if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
fi

echo "================================================"
echo "  TensorBoard"
echo "  Runs dir : $RUNS_DIR"
echo "  Filter   : ${FILTER:-all runs}"
echo "  URL      : http://localhost:${PORT}"
echo "  Remote   : http://$(hostname -I | awk '{print $1}'):${PORT}"
echo "  Runs available:"
ls "$RUNS_DIR" | sed 's/^/    /'
echo "  Stop     : Ctrl-C"
echo "================================================"

tensorboard \
    --logdir "$LOGDIR" \
    --port "$PORT" \
    --bind_all \
    --reload_interval 5
