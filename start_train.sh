#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_ROOT_DIR="${SCRIPT_DIR}/tmp"
TRAIN_LOG="${TMP_ROOT_DIR}/moe_train.log"
TRAIN_PID_FILE="${TMP_ROOT_DIR}/moe_train.pid"

mkdir -p "${TMP_ROOT_DIR}"

# Kill old processes
pkill -f "python.*moe.py" 2>/dev/null
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

# Start training in background
cd "${SCRIPT_DIR}"
nohup "${PY_CMD[@]}" -u moe.py --stage train --epochs 1 > "${TRAIN_LOG}" 2>&1 &
echo $! > "${TRAIN_PID_FILE}"

echo "Training started. Check ${TRAIN_LOG} for output"
echo "PID: $(cat "${TRAIN_PID_FILE}")"
