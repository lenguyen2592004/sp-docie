#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_ROOT_DIR="${WORKSPACE_DIR}/tmp"
DGL_SRC_DIR="${TMP_ROOT_DIR}/dgl-src/dgl"
BUILD_DIR="${DGL_SRC_DIR}/build_cuda128"
LEGACY_BUILD_DIR="${DGL_SRC_DIR}/build"
STEP6_LOG="${TMP_ROOT_DIR}/dgl_step6_install.log"
STEP7_LOG="${TMP_ROOT_DIR}/dgl_step7_validate.log"

PYTHON_BIN="$(which python)"
PIP_BIN="$(which pip)"
SITE_PACKAGES="$(${PYTHON_BIN} -c "import site; print(site.getsitepackages()[0])")"

mkdir -p "${TMP_ROOT_DIR}"

echo "[6/7] Install Python package"
if [[ ! -f "${BUILD_DIR}/libdgl.so" ]]; then
  echo "[ERROR] Missing ${BUILD_DIR}/libdgl.so. Please run build step 5 first."
  exit 1
fi

# setup.py in DGL python folder often searches ../build for -ldgl.
ln -sfn "${BUILD_DIR}" "${LEGACY_BUILD_DIR}"

# Ensure optional dependency exists so importing DGL data modules does not fail.
if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import pandas
PY
then
  "${PIP_BIN}" install pandas >"${TMP_ROOT_DIR}/dgl_step7_pandas_install.log" 2>&1
fi

cd "${DGL_SRC_DIR}/python"
"${PIP_BIN}" uninstall -y dgl || true
DGL_LIBRARY_PATH="${LEGACY_BUILD_DIR}" "${PYTHON_BIN}" setup.py install >"${STEP6_LOG}" 2>&1
cp -f "${BUILD_DIR}/libdgl.so" "${SITE_PACKAGES}/dgl/libdgl.so"

echo "[7/7] Validate CUDA backend"
cd "${WORKSPACE_DIR}"
PYTHONPATH="" "${PYTHON_BIN}" - <<'PY' >"${STEP7_LOG}" 2>&1
import torch
import pandas as pd
import dgl

print('pandas:', pd.__version__)
print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'available:', torch.cuda.is_available())
print('dgl:', dgl.__version__)
print('dgl_file:', dgl.__file__)
g = dgl.graph(([0], [0]), num_nodes=1)
g = g.to('cuda')
print('dgl graph device:', g.device)
print('DGL CUDA OK')
PY

echo "Done"
echo "Step 6 log: ${STEP6_LOG}"
echo "Step 7 log: ${STEP7_LOG}"
