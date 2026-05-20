#!/usr/bin/env bash
set -euo pipefail

# Commands used to diagnose and verify the DGL GraphBolt datapipe/pandas import bug fix.
cd /workspace

# 1) Reproduce the original failure
python -u moe.py \
  --stage train \
  --epochs 20 \
  --full-train \
  --full-eval \
  --full-test \
  --result-dir inference_results \
  --result-file result.json

# 2) Inspect relevant import and fallback code paths
rg -n "torchdata.datapipes|No module named 'pandas'|_safe_import_dgl|GraphBolt|DGL is required" \
  moe.py build_dgl_cuda128.sh moe.bash setup_all.sh

# 3) Validate syntax after patching
python -m py_compile moe.py
bash -n build_dgl_cuda128.sh
bash -n moe.bash
bash -n setup_all.sh

# 4) Quick lightweight startup check (avoid full 8B run)
python -u moe.py \
  --stage train \
  --epochs 1 \
  --train-limit 1 \
  --eval-limit 1 \
  --test-limit 1 \
  --model-id sshleifer/tiny-gpt2 \
  --cpu-debug-model sshleifer/tiny-gpt2 \
  --device cpu \
  --no-wandb \
  --result-dir inference_results \
  --result-file smoke_fix_datapipe.json \
  --debug

# 5) Review code changes
git --no-pager diff -- moe.py build_dgl_cuda128.sh moe.bash setup_all.sh fix_datapipe
