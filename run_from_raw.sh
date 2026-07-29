#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${VIDEO_ROOT:?Set VIDEO_ROOT to the directory containing V3C1 and V3C2}"
: "${TASKS_PATH:?Set TASKS_PATH to the task JSONL file}"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/models}"
SHOT_OUTPUT="${SHOT_OUTPUT:-${ARTIFACT_ROOT}/omnishot_output}"
V3C1_ROOT="${V3C1_ROOT:-${VIDEO_ROOT}/V3C1}"
V3C2_ROOT="${V3C2_ROOT:-${VIDEO_ROOT}/V3C2}"
OMNISHOT_PYTHON="${OMNISHOT_PYTHON:-${PROJECT_ROOT}/.venv-omnishot/bin/python}"
OMNISHOT_CHECKPOINT="${OMNISHOT_CHECKPOINT:-${MODEL_ROOT}/OmniShotCut/OmniShotCut_ckpt.pth}"
TORCH_HOME="${TORCH_HOME:-${MODEL_ROOT}/torch}"

if [[ ! -x "${OMNISHOT_PYTHON}" ]]; then
  echo "OmniShotCut environment not found: ${OMNISHOT_PYTHON}" >&2
  echo "Run scripts/setup_omnishot_env.sh first." >&2
  exit 1
fi
if [[ ! -f "${OMNISHOT_CHECKPOINT}" ]]; then
  echo "OmniShotCut checkpoint not found: ${OMNISHOT_CHECKPOINT}" >&2
  echo "Run scripts/download_omnishot_models.sh first." >&2
  exit 1
fi

export TORCH_HOME

"${OMNISHOT_PYTHON}" "${PROJECT_ROOT}/tools/omnishot_pipeline.py" shots \
  --dataset-root "${V3C1_ROOT}" \
  --dataset-root "${V3C2_ROOT}" \
  --out "${SHOT_OUTPUT}" \
  --omni-checkpoint "${OMNISHOT_CHECKPOINT}" \
  --omnishotcut-commit 3331cd3163f7f17cd6d7c8fc12ffde22894ace01

export SHOTS_DIR="${SHOT_OUTPUT}/shots"
export ARTIFACT_ROOT
export MODEL_ROOT

exec "${PROJECT_ROOT}/run_pipeline.sh"
