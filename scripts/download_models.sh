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
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/models}"

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found. Set PIPELINE_PYTHON to the environment Python." >&2
  exit 1
fi

mkdir -p \
  "${MODEL_ROOT}/Qwen3-VL-Embedding-8B" \
  "${MODEL_ROOT}/Qwen3-VL-Reranker-8B"

# Fetch the small tokenizer/config files from the pinned Hugging Face revisions.
# Weight files are fetched below from ModelScope because that endpoint is much
# faster and its ETags match the Hugging Face SHA-256 hashes.
"${PYTHON_BIN}" - "${MODEL_ROOT}" <<'PY'
from pathlib import Path
import sys

from huggingface_hub import snapshot_download

model_root = Path(sys.argv[1])
models = (
    (
        "Qwen/Qwen3-VL-Embedding-8B",
        "2c4565515e0f265c6511776e7193b22c0968ddc7",
    ),
    (
        "Qwen/Qwen3-VL-Reranker-8B",
        "b212dc8c91a8164aef1ea2de9c1a867611e75c04",
    ),
)
for model_id, revision in models:
    target = model_root / model_id.rsplit("/", 1)[-1]
    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=target,
        ignore_patterns=["*.safetensors"],
        max_workers=8,
    )
PY

fetch_shard() {
  model_name="$1"
  file_name="$2"
  expected_size="$3"
  expected_hash="$4"
  target_dir="${MODEL_ROOT}/${model_name}"
  final_path="${target_dir}/${file_name}"
  part_path="${final_path}.modelscope.part"

  if [[ -f "${final_path}" ]]; then
    actual_size="$(stat -c %s "${final_path}")"
    if [[ "${actual_size}" != "${expected_size}" ]]; then
      echo "Existing file has wrong size: ${final_path}" >&2
      return 1
    fi
    actual_hash="$(sha256sum "${final_path}" | awk '{print $1}')"
    if [[ "${actual_hash}" != "${expected_hash}" ]]; then
      echo "Existing file has wrong SHA-256: ${final_path}" >&2
      return 1
    fi
    echo "verified existing ${model_name}/${file_name}"
    return 0
  fi

  echo "downloading ${model_name}/${file_name}"
  curl \
    -L \
    --fail \
    --silent \
    --show-error \
    --retry 10 \
    --retry-all-errors \
    --continue-at - \
    --output "${part_path}" \
    "https://modelscope.cn/models/Qwen/${model_name}/resolve/master/${file_name}"

  actual_size="$(stat -c %s "${part_path}")"
  if [[ "${actual_size}" != "${expected_size}" ]]; then
    echo "Size mismatch: ${part_path}" >&2
    return 1
  fi
  actual_hash="$(sha256sum "${part_path}" | awk '{print $1}')"
  if [[ "${actual_hash}" != "${expected_hash}" ]]; then
    echo "SHA-256 mismatch: ${part_path}" >&2
    return 1
  fi
  mv "${part_path}" "${final_path}"
  echo "verified ${model_name}/${file_name}"
}

fetch_shard Qwen3-VL-Embedding-8B model-00001-of-00004.safetensors 4998056552 79ef275ec5f751d5fb59357c00d473268f9fd74abf5e38aa30137d268e7733c4 &
pid_1=$!
fetch_shard Qwen3-VL-Embedding-8B model-00002-of-00004.safetensors 4915962464 a4da61f512e84fc0f0b80bcb7bcc5137eb3bf25b658a7d55f84f4056078545f0 &
pid_2=$!
fetch_shard Qwen3-VL-Embedding-8B model-00003-of-00004.safetensors 4915962496 7fb17cf8f06d6fe5aaacf114e85c4e6d8318799f24b517f50d1ec154a8d47007 &
pid_3=$!
fetch_shard Qwen3-VL-Embedding-8B model-00004-of-00004.safetensors 1459698112 000213b6d1d03ed9023fac23716da51ec4c5be221a04526c2f732d31d8fed1f5 &
pid_4=$!
fetch_shard Qwen3-VL-Reranker-8B model-00001-of-00004.safetensors 4998056552 00db2779f4c81c18a551b05ee617a4012af2601ec47181c15629ae756ef367d6 &
pid_5=$!
fetch_shard Qwen3-VL-Reranker-8B model-00002-of-00004.safetensors 4915962464 6ef4ccabf1f72c42eed016adca6d46528e66875171268e5ff603c1df2f97fa3d &
pid_6=$!
fetch_shard Qwen3-VL-Reranker-8B model-00003-of-00004.safetensors 4915962496 f036e887d2b27b56a3b22fe200bbdd7e0c8f075100f87bd2b6653fa8fca02973 &
pid_7=$!
fetch_shard Qwen3-VL-Reranker-8B model-00004-of-00004.safetensors 2704357976 723d600bc3947051da769a35fb7bb62419d60d558bee7e9c9bb15919ccc79190 &
pid_8=$!

wait \
  "${pid_1}" \
  "${pid_2}" \
  "${pid_3}" \
  "${pid_4}" \
  "${pid_5}" \
  "${pid_6}" \
  "${pid_7}" \
  "${pid_8}"

find \
  "${MODEL_ROOT}/Qwen3-VL-Embedding-8B/.cache/huggingface/download" \
  "${MODEL_ROOT}/Qwen3-VL-Reranker-8B/.cache/huggingface/download" \
  -type f \
  -name "*.incomplete" \
  -delete

echo "Both Qwen3-VL 8B snapshots are complete and verified."
