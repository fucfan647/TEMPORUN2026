#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${OMNISHOT_ENV_DIR:-${PROJECT_ROOT}/.venv-omnishot}"
PYTHON_BIN="${OMNISHOT_PYTHON:-${ENV_DIR}/bin/python}"
OMNISHOT_REPO="${PROJECT_ROOT}/vendor/OmniShotCut"
OMNISHOT_COMMIT="3331cd3163f7f17cd6d7c8fc12ffde22894ace01"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  uv venv --python 3.10 "${ENV_DIR}"
fi

uv pip install --python "${PYTHON_BIN}" \
  torch==2.3.1 torchvision==0.18.1 \
  --index-url https://download.pytorch.org/whl/cu121
uv pip install --python "${PYTHON_BIN}" \
  -r "${PROJECT_ROOT}/requirements-omnishot.txt"

if [[ ! -d "${OMNISHOT_REPO}/.git" ]]; then
  mkdir -p "${PROJECT_ROOT}/vendor"
  git clone https://github.com/UVA-Computer-Vision-Lab/OmniShotCut.git \
    "${OMNISHOT_REPO}"
fi

git -C "${OMNISHOT_REPO}" fetch --depth 1 origin "${OMNISHOT_COMMIT}"
git -C "${OMNISHOT_REPO}" checkout --detach "${OMNISHOT_COMMIT}"
uv pip install --python "${PYTHON_BIN}" --no-deps "${OMNISHOT_REPO}"

"${PYTHON_BIN}" - <<'PY'
import importlib.metadata as metadata
import torch
import torchvision

print("omnishotcut", metadata.version("omnishotcut"))
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("torchvision", torchvision.__version__)
print("cuda_available", torch.cuda.is_available())
PY
