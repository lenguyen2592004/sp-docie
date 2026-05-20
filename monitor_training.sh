#!/bin/bash
# Monitor training progress by watching wandb runs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_ROOT_DIR="${SCRIPT_DIR}/tmp"
TRAIN_LOG="${TMP_ROOT_DIR}/train_latest.log"

mkdir -p "${TMP_ROOT_DIR}"

echo "=== Training Monitor ==="
echo "Watching for new wandb runs..."
echo ""

cd "${SCRIPT_DIR}"

# Start training in background
nohup python moe.py --stage train --epochs 1 > "${TRAIN_LOG}" 2>&1 &
TRAIN_PID=$!
echo "Training started with PID: $TRAIN_PID"
echo ""

# Monitor for 120 seconds
for i in {1..120}; do
    sleep 1
    # Check if process still running
    if ! ps -p $TRAIN_PID > /dev/null 2>&1; then
        echo "Training process completed or crashed at $(date)"
        break
    fi
    
    # Show progress every 10 seconds
    if [ $((i % 10)) -eq 0 ]; then
        echo "[$i/120s] Training running..."
        # Show last few lines of log
        if [ -f "${TRAIN_LOG}" ]; then
            echo "--- Last 3 lines ---"
            tail -n 3 "${TRAIN_LOG}"
        fi
    fi
done

echo ""
echo "=== Final Status ==="
if [ -f "${TRAIN_LOG}" ]; then
    echo "Last 20 lines of training log:"
    tail -n 20 "${TRAIN_LOG}"
fi

# Check for errors
if grep -i "error\|traceback\|exception" "${TRAIN_LOG}"; then
    echo ""
    echo "❌ Errors detected in training log"
    exit 1
else
    echo ""
    echo "✓ No errors found in log"
    exit 0
fi
