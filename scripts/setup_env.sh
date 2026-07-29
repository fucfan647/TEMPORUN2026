#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${PIPELINE_PYTHON:-}" ]]; then
  PYTHON_BIN="${PIPELINE_PYTHON}"
elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
else
  PYTHON_BIN="$(command -v python3 || true)"
fi
QWEN_REPO="${PROJECT_ROOT}/vendor/Qwen3-VL-Embedding"
QWEN_COMMIT="393e2978d27852b0d0230d6994f37f9c15bed73c"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found at ${PYTHON_BIN}" >&2
  exit 1
fi

QWEN_EMBEDDING_SOURCE="${QWEN_REPO}/src/models/qwen3_vl_embedding.py"
QWEN_RERANKER_SOURCE="${QWEN_REPO}/src/models/qwen3_vl_reranker.py"

if [[ -f "${QWEN_EMBEDDING_SOURCE}" && -f "${QWEN_RERANKER_SOURCE}" \
      && ! -d "${QWEN_REPO}/.git" ]]; then
  echo "Using Qwen source vendored in the pipeline archive."
elif [[ ! -d "${QWEN_REPO}/.git" ]]; then
  rm -rf "${QWEN_REPO}"
  mkdir -p "${PROJECT_ROOT}/vendor"
  git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git "${QWEN_REPO}"
fi

if [[ -d "${QWEN_REPO}/.git" ]]; then
  git -C "${QWEN_REPO}" fetch --depth 1 origin "${QWEN_COMMIT}"
  git -C "${QWEN_REPO}" checkout --detach "${QWEN_COMMIT}"
fi

uv pip install --python "${PYTHON_BIN}" -r "${PROJECT_ROOT}/requirements.txt"
uv pip install --python "${PYTHON_BIN}" --no-deps -e "${PROJECT_ROOT}"

"${PYTHON_BIN}" -m temporun_pipeline doctor --qwen-source "${QWEN_REPO}"
