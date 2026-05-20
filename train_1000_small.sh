#!/usr/bin/env bash
set -euo pipefail

# Train 1000 epochs on small subsets (train/val/test) using moe.py
# Defaults:
#   epochs=1000
#   train=20 docs from train_annotated.json
#   val=10 docs split from train_annotated.json (non-overlapping)
#   test=10 docs from dev.json
#
# Usage examples:
#   ./train_1000_small.sh --no-wandb
#   ./train_1000_small.sh --epochs 1000 --train 20 --val 10 --test 10 --no-wandb
#   ./train_1000_small.sh --test 10 --no-wandb
#   ./train_1000_small.sh --model-id sshleifer/tiny-gpt2 --debug --epochs 3

EPOCHS=1000
TRAIN=20
VAL=10
TEST=10

TRAIN_FILE="train_annotated.json"
VAL_FILE="train_annotated.json"
TEST_FILE="dev.json"

EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --epochs)
      EPOCHS="$2"; shift 2 ;;
    --train)
      TRAIN="$2"; shift 2 ;;
    --val)
      VAL="$2"; shift 2 ;;
    --test)
      TEST="$2"; shift 2 ;;

    --train-file)
      TRAIN_FILE="$2"; shift 2 ;;
    --val-file|--eval-file)
      VAL_FILE="$2"; shift 2 ;;
    --test-file)
      TEST_FILE="$2"; shift 2 ;;
    --no-test)
      TEST_FILE=""; shift ;;

    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

cd /workspace

CMD=(python -u moe.py \
  --stage train \
  --device cuda \
  --epochs "$EPOCHS" \
  --patience "$EPOCHS" \
  --train-file "$TRAIN_FILE" \
  --eval-file "$VAL_FILE" \
  --train-limit "$TRAIN" \
  --eval-limit "$VAL")

# Make validation a split from the training file (non-overlapping):
# train: [0..TRAIN)
# val:   [TRAIN..TRAIN+VAL)
CMD+=(--eval-from-train --eval-offset "$TRAIN")

if [[ -n "${TEST_FILE}" ]]; then
  if [[ -f "${TEST_FILE}" ]]; then
    CMD+=(--test-file "$TEST_FILE" --test-limit "$TEST")
  else
    echo "[WARN] Test file not found: ${TEST_FILE} (skipping final test eval)" >&2
  fi
fi

CMD+=("${EXTRA_ARGS[@]}")

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
