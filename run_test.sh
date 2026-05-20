#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_ROOT_DIR="${SCRIPT_DIR}/tmp"
LOG_FILE="${TMP_ROOT_DIR}/train_test.log"

mkdir -p "${TMP_ROOT_DIR}"

# Kill old processes
pkill -9 -f "python.*moe.py" 2>/dev/null
sleep 3

PY_CMD=(python)
if command -v conda >/dev/null 2>&1; then
    if conda env list | awk '{print $1}' | grep -qx "moe"; then
        if conda run -n moe python -c "import dgl" >/dev/null 2>&1; then
            PY_CMD=(conda run -n moe python)
        else
            echo "[WARN] conda env 'moe' exists but cannot import dgl; using current python."
        fi
    fi
fi

# Start lightweight smoke training
cd "${SCRIPT_DIR}"
nohup "${PY_CMD[@]}" -u moe.py \
    --stage train \
    --debug \
    --debug-train-samples 4 \
    --debug-samples 2 \
    --epochs 1 \
    --device cpu \
    --no-wandb \
    > "${LOG_FILE}" 2>&1 &
PID=$!
echo "Started training PID: $PID"

    # Timeboxed smoke window to catch early runtime crashes.
    SMOKE_WAIT_SEC="${SMOKE_WAIT_SEC:-30}"
    sleep "${SMOKE_WAIT_SEC}"

    if kill -0 "$PID" >/dev/null 2>&1; then
        echo "Process still running after ${SMOKE_WAIT_SEC}s; this is acceptable for smoke test."
    else
        echo "Process ended within ${SMOKE_WAIT_SEC}s."
    fi

# Show log
echo ""
echo "=== Training Log ==="
tail -50 "${LOG_FILE}"

# Check for crash signatures only (avoid false positives from warning text).
if grep -qE "Traceback \(most recent call last\)|RuntimeError:|ValueError:|ImportError:|ModuleNotFoundError:" "${LOG_FILE}"; then
    grep -E "Traceback \(most recent call last\)|RuntimeError:|ValueError:|ImportError:|ModuleNotFoundError:" "${LOG_FILE}" | head -10
    echo ""
    echo "❌ Errors detected"
    exit 1
else
    echo ""
    echo "✓ No errors"
    exit 0
fi
