#!/bin/bash
set -euo pipefail

pip install --upgrade pip
pip install torch torchvision torchdata peft bitsandbytes accelerate transformers
pip install scikit-learn wandb pandas
pip install lightning pytorch-lightning
pip install pot

# Install DGL matching current torch/cuda when possible.
TORCH_MM=$(python - <<'PY'
import torch
v = torch.__version__.split('+')[0].split('.')
print(f"{v[0]}.{v[1]}")
PY
)

CUDA_TAG=$(python - <<'PY'
import torch
v = torch.version.cuda
print(f"cu{v.replace('.', '')}" if v else "cpu")
PY
)

echo "[DGL] torch=${TORCH_MM} cuda_tag=${CUDA_TAG}"
if [[ "${CUDA_TAG}" == "cpu" ]]; then
	pip install dgl -f "https://data.dgl.ai/wheels/torch-${TORCH_MM}/repo.html" || pip install dgl
else
	pip install dgl -f "https://data.dgl.ai/wheels/torch-${TORCH_MM}/${CUDA_TAG}/repo.html" \
		|| pip install dgl -f "https://data.dgl.ai/wheels/torch-${TORCH_MM}/repo.html" \
		|| pip install dgl
fi

# pip install triton-nightly # Optional
