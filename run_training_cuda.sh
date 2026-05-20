#!/usr/bin/env bash
set -euo pipefail

# Quick training run to validate DGL CUDA integration
# Uses small subset for fast iteration

echo "============================================================"
echo "Starting MoE Training with DGL CUDA Backend"
echo "============================================================"

cd /workspace/moe-fix-26-02-5090-bug-nan

# Configuration
EPOCHS=3
TRAIN_DOCS=10
VAL_DOCS=5
MODEL="sshleifer/tiny-gpt2"  # Small model for quick validation
NUM_GPUS="${NUM_GPUS:-2}"

# Kill any existing training
pkill -f "python.*moe.py" 2>/dev/null || true
sleep 2

# Launch training
echo ""
echo "Configuration:"
echo "  Model: $MODEL"
echo "  Device: CUDA (with DGL CUDA backend)"
echo "  GPUs: $NUM_GPUS (DDP via torchrun)"
echo "  Epochs: $EPOCHS"
echo "  Train docs: $TRAIN_DOCS"
echo "  Val docs: $VAL_DOCS"
echo ""
echo "Command:"

CMD=(
    /venv/moe/bin/torchrun
    --standalone
    --nproc_per_node "$NUM_GPUS"
    moe.py
    --stage train
    --device cuda
    --model-id "$MODEL"
    --epochs "$EPOCHS"
    --patience "$EPOCHS"
    --train-file train_annotated.json
    --eval-file train_annotated.json
    --eval-from-train
    --train-limit "$TRAIN_DOCS"
    --eval-limit "$VAL_DOCS"
    --eval-offset "$TRAIN_DOCS"
    --batch-size 2
    --num-experts 4
    --no-wandb
)

echo "${CMD[*]}"
echo ""
echo "============================================================"
echo "Training Output:"
echo "============================================================"
echo ""

exec "${CMD[@]}"
