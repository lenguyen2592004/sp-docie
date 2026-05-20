#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_ROOT_DIR="${SCRIPT_DIR}/tmp"

# Runtime-only overrides (separate from setup scripts)
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

export HF_HOME="${HF_HOME:-${TMP_ROOT_DIR}/hf_home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

export WANDB__SERVICE_WAIT="300"
export WANDB_CONSOLE="wrap"
export WANDB_DISABLE_GIT="true"
export WANDB_MODE="online"
if [[ "${WANDB_DISABLED:-}" =~ ^(1|true|yes)$ ]]; then
	unset WANDB_DISABLED
fi
export DGL_DISABLE_GRAPHBOLT="1"

mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${PWD}/inference_results" "${PWD}/checkpoints"

echo "[ENV] Runtime overrides active"
echo "[ENV] HF_HOME=${HF_HOME}"
echo "[ENV] PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "[ENV] WANDB_MODE=${WANDB_MODE}"
echo "[ENV] DGL_DISABLE_GRAPHBOLT=${DGL_DISABLE_GRAPHBOLT}"
