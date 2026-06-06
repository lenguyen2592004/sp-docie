#!/usr/bin/env bash
set -euo pipefail
# Use a conservative default branch/tag for compatibility; can be overridden.
DGL_VERSION_TAG="${DGL_VERSION_TAG:-v2.1.0}"
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_ROOT_DIR="${WORKSPACE_DIR}/tmp"
DGL_SRC_ROOT="${TMP_ROOT_DIR}/dgl-src"
DGL_SRC_DIR="${DGL_SRC_ROOT}/dgl"
BUILD_DIR_SUFFIX="${DGL_BUILD_SUFFIX:-cuda}"
LOG_FILE="${TMP_ROOT_DIR}/dgl_build_cuda128.log"
# Thay đổi đường dẫn cứng thành path động
PYTHON_BIN=$(which python)
PIP_BIN=$(which pip)
PYTHON_BIN_DIR=$(dirname "${PYTHON_BIN}")
# Lấy site-packages path tự động
SITE_PACKAGES=$($PYTHON_BIN -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
BUILD_JOBS="${DGL_BUILD_JOBS:-1}"
FORCE_SOURCE_BUILD="${FORCE_DGL_SOURCE_BUILD:-0}"

mkdir -p "${TMP_ROOT_DIR}"

patch_graphbolt_for_torchdata() {
  "${PYTHON_BIN}" - <<'PY'
import os
import site

shim = '''"""Graphbolt compatibility shim for environments without optional deps."""
import warnings
try:
  from .base import *
  from .minibatch import *
  from .dataloader import *
  from .dataset import *
  from .feature_fetcher import *
  from .feature_store import *
  from .impl import *
  from .itemset import *
  from .item_sampler import *
  from .minibatch_transformer import *
  from .negative_sampler import *
  from .sampled_subgraph import *
  from .subgraph_sampler import *
except ModuleNotFoundError as exc:
  msg = str(exc)
  if ('torchdata.datapipes' in msg) or ("No module named 'pandas'" in msg) or ('No module named "pandas"' in msg):
    warnings.warn(f'GraphBolt disabled: missing optional dependency ({msg}).')
  else:
    raise
'''

paths = []
try:
  paths.extend(site.getsitepackages())
except Exception:
  pass
try:
  user_site = site.getusersitepackages()
  if isinstance(user_site, str):
    paths.append(user_site)
except Exception:
  pass

patched = False
for base in paths:
  p = os.path.join(base, 'dgl', 'graphbolt', '__init__.py')
  if not os.path.isfile(p):
    continue
  with open(p, 'r', encoding='utf-8') as f:
    txt = f.read()
  if ('compatibility shim for environments without optional deps' in txt) or ('compatibility shim for environments without torchdata.datapipes' in txt):
    patched = True
    continue
  with open(p, 'w', encoding='utf-8') as f:
    f.write(shim)
  patched = True

print('[INFO] GraphBolt torchdata compatibility patch:', 'applied' if patched else 'not-needed')
PY
}

detect_cuda_home() {
  if [ -n "${CUDA_HOME:-}" ] && [ -x "${CUDA_HOME}/bin/nvcc" ]; then
    echo "${CUDA_HOME}"
    return
  fi
  if [ -n "${CUDA_PATH:-}" ] && [ -x "${CUDA_PATH}/bin/nvcc" ]; then
    echo "${CUDA_PATH}"
    return
  fi
  if command -v nvcc >/dev/null 2>&1; then
    local nvcc_bin
    nvcc_bin=$(command -v nvcc)
    echo "$(cd "$(dirname "${nvcc_bin}")/.." && pwd)"
    return
  fi
  if [ -x "/usr/local/cuda/bin/nvcc" ]; then
    echo "/usr/local/cuda"
    return
  fi
  echo ""
}

CUDA_HOME_DETECTED="$(detect_cuda_home)"
if [ -z "${CUDA_HOME_DETECTED}" ]; then
  echo "[ERROR] Could not find nvcc. Set CUDA_HOME or ensure nvcc is on PATH."
  exit 1
fi
NVCC_BIN="${CUDA_HOME_DETECTED}/bin/nvcc"

detect_cuda_target() {
  local torch_cuda
  torch_cuda=$(
    "${PYTHON_BIN}" - <<'PY'
import torch
print((torch.version.cuda or '').strip())
PY
  )
  if [ -n "${torch_cuda}" ]; then
    echo "${torch_cuda}"
    return
  fi
  "${NVCC_BIN}" --version | awk -F'release ' '/release/{print $2}' | awk -F',' '{print $1}'
}

detect_cuda_arch() {
  if [ -n "${DGL_CUDA_ARCH:-}" ]; then
    echo "${DGL_CUDA_ARCH}"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local cc
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.')
    if [ -n "${cc}" ]; then
      echo "${cc}"
      return
    fi
  fi
  echo "90"
}

validate_dgl_cuda() {
  (
    cd "${WORKSPACE_DIR}"
    PYTHONPATH="" "${PYTHON_BIN}" - <<'PY'
import sys
import torch
import dgl

if not torch.cuda.is_available():
    sys.exit(1)

g = dgl.graph(([0], [0]), num_nodes=1).to('cuda')
if str(g.device).startswith('cuda'):
    sys.exit(0)
sys.exit(1)
PY
  )
}

validate_dgl_import() {
  (
    cd "${WORKSPACE_DIR}"
    PYTHONPATH="" "${PYTHON_BIN}" - <<'PY'
import dgl
print(dgl.__version__)
PY
  )
}

CUDA_TARGET="$(detect_cuda_target)"
CUDA_ARCH_BIN="$(detect_cuda_arch)"
BUILD_DIR="${DGL_SRC_DIR}/build_${BUILD_DIR_SUFFIX}_${CUDA_TARGET//./}"

restore_cpu_dgl_on_failure() {
  if (cd "${WORKSPACE_DIR}" && PYTHONPATH="" "${PYTHON_BIN}" -c "import dgl") >/dev/null 2>&1; then
    return 0
  fi
  echo "[WARN] DGL import is broken after failed build; restoring wheel dgl==2.1.0 (CPU fallback)."
  "${PIP_BIN}" install --no-deps dgl==2.1.0 >/dev/null 2>&1 || true
    patch_graphbolt_for_torchdata || true
}

trap 'restore_cpu_dgl_on_failure' ERR

patch_graphbolt_for_torchdata || true

if [ "${FORCE_SOURCE_BUILD}" != "1" ] && validate_dgl_import >/dev/null 2>&1 && validate_dgl_cuda >/dev/null 2>&1; then
  echo "[SKIP] Existing DGL import + CUDA backend are healthy. Set FORCE_DGL_SOURCE_BUILD=1 to rebuild."
  exit 0
fi

if [ "${FORCE_SOURCE_BUILD}" != "1" ] && validate_dgl_cuda >/dev/null 2>&1; then
  echo "[SKIP] Existing DGL build already supports CUDA. Set FORCE_DGL_SOURCE_BUILD=1 to rebuild."
  exit 0
fi

echo "[1/7] Check CUDA toolkit"
"${NVCC_BIN}" --version
echo "[INFO] torch CUDA target: ${CUDA_TARGET}"
echo "[INFO] CUDA arch bin: ${CUDA_ARCH_BIN}"
echo "[INFO] CUDA_HOME: ${CUDA_HOME_DETECTED}"

echo "[2/7] Prepare Python build deps"
"${PIP_BIN}" install -U pip setuptools wheel "cython<3" "cmake<4" ninja pandas

echo "[3/7] Fetch DGL source (${DGL_VERSION_TAG})"
mkdir -p "${DGL_SRC_ROOT}"
if [ ! -d "${DGL_SRC_DIR}/.git" ]; then
  rm -rf "${DGL_SRC_DIR}"
  git clone --recursive https://github.com/dmlc/dgl.git "${DGL_SRC_DIR}"
fi
cd "${DGL_SRC_DIR}"
git fetch --tags
git checkout -f "${DGL_VERSION_TAG}"
git submodule update --init --recursive

# GCC 13 + DGL v2.1.0 may fail at src/runtime/dlpack_convert.cc because
# std::uintptr_t is used without including <cstdint>.
DL_CONVERT_FILE="${DGL_SRC_DIR}/src/runtime/dlpack_convert.cc"
if ! grep -q "#include <cstdint>" "${DL_CONVERT_FILE}"; then
  echo "[PATCH] Adding missing <cstdint> include to dlpack_convert.cc"
  awk '
    /#include <dlpack\/dlpack.h>/ {
      print
      print "#include <cstdint>"
      next
    }
    { print }
  ' "${DL_CONVERT_FILE}" > "${DL_CONVERT_FILE}.tmp"
  mv "${DL_CONVERT_FILE}.tmp" "${DL_CONVERT_FILE}"
fi

echo "[4/7] Configure CMake for CUDA ${CUDA_TARGET} (sm_${CUDA_ARCH_BIN})"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
export PATH="${PYTHON_BIN_DIR}:${CUDA_HOME_DETECTED}/bin:${PATH}"
export CUDA_HOME="${CUDA_HOME_DETECTED}"
export CUDACXX="${NVCC_BIN}"

cmake -G Ninja .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_C_FLAGS="-Wno-error -Wno-error=maybe-uninitialized" \
  -DCMAKE_CXX_FLAGS="-Wno-error" \
  -DUSE_CUDA=ON \
  -DUSE_GRAPHBOLT=OFF \
  -DUSE_OPENMP=ON \
  -DUSE_LIBXSMM=OFF \
  -DPYTHON_EXECUTABLE="${PYTHON_BIN}" \
  -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME_DETECTED}" \
  -DCUDA_ARCH_NAME=Manual \
  -DCUDA_ARCH_BIN="${CUDA_ARCH_BIN}" \
  -DCUDA_ARCH_PTX="${CUDA_ARCH_BIN}" \
  -DBUILD_CPP_TEST=OFF \
  > "${LOG_FILE}" 2>&1

echo "[5/7] Build libdgl (see ${LOG_FILE})"
cmake --build . -j"${BUILD_JOBS}" >> "${LOG_FILE}" 2>&1

echo "[6/7] Install Python package into /venv/moe"
cd "${DGL_SRC_DIR}/python"
DGL_LIBRARY_PATH="${BUILD_DIR}" "${PIP_BIN}" install -v . >> "${LOG_FILE}" 2>&1
cp -f "${BUILD_DIR}/libdgl.so" "${SITE_PACKAGES}/dgl/libdgl.so"
patch_graphbolt_for_torchdata || true

echo "[7/7] Validate CUDA backend"
cd "${WORKSPACE_DIR}"
PYTHONPATH="" "${PYTHON_BIN}" - <<'PY'
import torch
import dgl
print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'available:', torch.cuda.is_available())
print('dgl:', dgl.__version__)
if not torch.cuda.is_available():
  raise SystemExit('CUDA is not available in torch; cannot validate DGL CUDA backend.')
g = dgl.graph(([0], [0]), num_nodes=1)
g = g.to('cuda')
print('dgl graph device:', g.device)
print('DGL CUDA OK')
PY

trap - ERR

echo "Done. Build log: ${LOG_FILE}"
