#!/bin/bash
# ========================================================================================
# ONE-STOP SETUP SCRIPT FOR DOCRE MOE PROJECT
# Combines: setup_moe_env.sh, moe.bash, build_dgl_cuda128.sh, and runtime patches.
# ========================================================================================
set -euo pipefail

patch_graphbolt_if_needed() {
  python - <<'PY'
import os
import pathlib

try:
    import dgl
except Exception:
    raise SystemExit(0)

p = pathlib.Path(os.path.dirname(dgl.__file__)) / 'graphbolt' / '__init__.py'
if not p.exists():
    raise SystemExit(0)

text = p.read_text()
if 'warnings.warn(f"GraphBolt disabled: {exc}")' in text:
    print('GraphBolt patch already present.')
    raise SystemExit(0)

if text.rstrip().endswith('load_graphbolt()'):
    patched = text.rstrip()[:-len('load_graphbolt()')] + (
        'try:\n'
        '    load_graphbolt()\n'
        'except (FileNotFoundError, ImportError) as exc:\n'
        '    import warnings\n'
        '    warnings.warn(f"GraphBolt disabled: {exc}")\n'
    )
    p.write_text(patched)
    print(f'Patched {p}')
else:
    print('GraphBolt call site not in expected form; skipped patch.')
PY
}

# 1. ENVIRONMENT CREATION (Conda)
echo "[1/6] Creating & activating conda environment 'moe'..."
conda create -n moe python=3.10 -y || echo "Environment 'moe' already exists or skipping..."
eval "$(conda shell.bash hook)"
conda activate moe

# 2. CORE DEPENDENCIES
echo "[2/6] Installing core machine learning libraries..."
pip install --upgrade pip
pip install torch torchvision torchdata peft bitsandbytes accelerate transformers
pip install scikit-learn wandb lightning pytorch-lightning pot
pip install numpy psutil requests scipy tqdm networkx pandas

# 3. DGL INSTALLATION (WHEEL FALLBACK)
echo "[3/6] Attempting DGL installation via wheels..."
TORCH_MM=$(python -c "import torch; v = torch.__version__.split('+')[0].split('.'); print(f'{v[0]}.{v[1]}')")
CUDA_TAG=$(python -c "import torch; v = torch.version.cuda; print(f'cu{v.replace(\".\", \"\")}' if v else \"cpu\")")

if [[ "${CUDA_TAG}" == "cpu" ]]; then
    pip install dgl -f "https://data.dgl.ai/wheels/torch-${TORCH_MM}/repo.html" || pip install dgl
else
    pip install dgl -f "https://data.dgl.ai/wheels/torch-${TORCH_MM}/${CUDA_TAG}/repo.html" \
        || pip install dgl -f "https://data.dgl.ai/wheels/torch-${TORCH_MM}/repo.html" \
        || pip install dgl
fi

# 4. PATCH GRAPHBOLT IMPORT (for torch 2.10 mismatch)
echo "[4/6] Applying GraphBolt compatibility patch when needed..."
patch_graphbolt_if_needed

# 5. ENSURE DGL CUDA BACKEND (build from source if wheel is CPU-only)
echo "[5/6] Verifying DGL CUDA backend..."
if python - <<'PY'
import torch
import dgl
if not torch.cuda.is_available():
    raise SystemExit(0)
g = dgl.graph(([0], [0]), num_nodes=1).to('cuda')
print('DGL CUDA OK', g.device)
PY
then
  echo "DGL CUDA backend ready."
else
  echo "DGL CUDA backend missing. Building DGL from source..."
    DGL_BUILD_JOBS=${DGL_BUILD_JOBS:-1} /bin/bash ./build_dgl_cuda128.sh
  patch_graphbolt_if_needed
fi

# 6. FINAL VALIDATION
echo "[6/6] Finalizing installation check..."
python - <<'PY'
import torch, dgl, transformers, peft
print('Environment Summary:')
print(f'- Torch: {torch.__version__} (CUDA available: {torch.cuda.is_available()})')
print(f'- DGL: {dgl.__version__}')
print(f'- Transformers: {transformers.__version__}')
if torch.cuda.is_available():
    try:
        g = dgl.graph(([0], [0]), num_nodes=1).to('cuda')
        print(f'- DGL graph device: {g.device}')
    except Exception as exc:
        print(f'- DGL CUDA check: FAIL ({exc})')
PY

echo "========================================================================================"
echo "Setup complete. Activate with: 'conda activate moe'"
echo "To run training: 'python moe.py --stage train --device cuda'"
echo "========================================================================================"
