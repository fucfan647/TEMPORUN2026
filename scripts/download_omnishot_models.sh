#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/models}"
PYTHON_BIN="${OMNISHOT_PYTHON:-${PROJECT_ROOT}/.venv-omnishot/bin/python}"
OMNI_DIR="${MODEL_ROOT}/OmniShotCut"
TORCH_HOME="${TORCH_HOME:-${MODEL_ROOT}/torch}"
OMNI_FILE="${OMNI_DIR}/OmniShotCut_ckpt.pth"
RESNET_FILE="${TORCH_HOME}/hub/checkpoints/resnet18-f37072fd.pth"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "OmniShotCut Python not found: ${PYTHON_BIN}" >&2
  echo "Run scripts/setup_omnishot_env.sh first." >&2
  exit 1
fi

mkdir -p "${OMNI_DIR}" "$(dirname "${RESNET_FILE}")"

"${PYTHON_BIN}" - "${OMNI_DIR}" <<'PY'
from pathlib import Path
import sys

from huggingface_hub import hf_hub_download

target = Path(sys.argv[1])
hf_hub_download(
    repo_id="uva-cv-lab/OmniShotCut",
    filename="OmniShotCut_ckpt.pth",
    revision="7f646c4ff4bb843e18c013481fb5d9ed2b068c6b",
    local_dir=target,
)
PY

verify_file() {
  local path="$1"
  local expected_size="$2"
  local expected_hash="$3"
  local actual_size
  local actual_hash

  actual_size="$(stat -Lc %s "${path}")"
  if [[ "${actual_size}" != "${expected_size}" ]]; then
    echo "Size mismatch: ${path}" >&2
    exit 1
  fi
  actual_hash="$(sha256sum "${path}" | awk '{print $1}')"
  if [[ "${actual_hash}" != "${expected_hash}" ]]; then
    echo "SHA-256 mismatch: ${path}" >&2
    exit 1
  fi
  echo "verified ${path}"
}

verify_file \
  "${OMNI_FILE}" \
  164149963 \
  5948ea78e00626c0e6c5e742e64873ef872cf4a5071d2a0841aed51c3e686cfa

if [[ ! -f "${RESNET_FILE}" ]]; then
  curl \
    -L \
    --fail \
    --silent \
    --show-error \
    --retry 10 \
    --retry-all-errors \
    --output "${RESNET_FILE}.part" \
    https://download.pytorch.org/models/resnet18-f37072fd.pth
  mv "${RESNET_FILE}.part" "${RESNET_FILE}"
fi

verify_file \
  "${RESNET_FILE}" \
  46830571 \
  f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec
