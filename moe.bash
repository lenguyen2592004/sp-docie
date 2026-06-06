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

# =====================================================================================
# RoBERTa-large backbone (Part 1) — NO new packages required.
# transformers + peft (installed above) are sufficient; bitsandbytes is now OPTIONAL
# (only used for 4-bit causal LMs like Qwen). The block below is OPTIONAL convenience:
# pre-download the encoder + sanity-check it loads as an encoder with the [E1]/[E2]
# markers, so the first training run doesn't stall on a network download.
# =====================================================================================
python - <<'PY'
from transformers import AutoConfig, AutoModel, AutoTokenizer
mid = "roberta-large"
cfg = AutoConfig.from_pretrained(mid)
assert not getattr(cfg, "is_decoder", False), "roberta-large should be an encoder"
tok = AutoTokenizer.from_pretrained(mid, add_prefix_space=True)
tok.add_special_tokens({"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]})
m = AutoModel.from_pretrained(mid)                 # downloads ~1.4GB on first run
m.resize_token_embeddings(len(tok))
print(f"[OK] {mid}: hidden_size={cfg.hidden_size}, max_pos={cfg.max_position_embeddings}, "
      f"pad={tok.pad_token!r}, E1_id={tok.convert_tokens_to_ids('[E1]')}")
PY

# Syntax-check the training script (does NOT run training).
python -m py_compile moe.py && echo "[OK] moe.py compiles"
