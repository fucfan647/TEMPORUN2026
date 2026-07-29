#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PIPELINE_PYTHON:-}" ]]; then
  PYTHON_BIN="${PIPELINE_PYTHON}"
elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
else
  PYTHON_BIN="$(command -v python3 || true)"
fi

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found. Set PIPELINE_PYTHON to the environment Python." >&2
  exit 1
fi

: "${TASKS_PATH:?Set TASKS_PATH to the task JSONL file}"
: "${SHOTS_DIR:?Set SHOTS_DIR to the shot JSON directory}"
: "${VIDEO_ROOT:?Set VIDEO_ROOT to the V3C video root}"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts}"
CORPUS_ROOT="${CORPUS_ROOT:-${ARTIFACT_ROOT}}"
TASK_OUTPUT_ROOT="${TASK_OUTPUT_ROOT:-${ARTIFACT_ROOT}}"
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/models}"
EMBED_MODEL="${EMBED_MODEL:-${MODEL_ROOT}/Qwen3-VL-Embedding-8B}"
RERANK_MODEL="${RERANK_MODEL:-${MODEL_ROOT}/Qwen3-VL-Reranker-8B}"
QWEN_SOURCE="${QWEN_SOURCE:-${PROJECT_ROOT}/vendor/Qwen3-VL-Embedding}"
REUSE_RETRIEVAL_PARTITIONS="${REUSE_RETRIEVAL_PARTITIONS:-false}"

mkdir -p "${CORPUS_ROOT}" "${TASK_OUTPUT_ROOT}"

child_pids=()

cleanup_children() {
  if ((${#child_pids[@]})); then
    kill "${child_pids[@]}" 2>/dev/null || true
    wait "${child_pids[@]}" 2>/dev/null || true
  fi
}

wait_for_children() {
  local status
  local finished_pid
  local pid
  local -a active=("${child_pids[@]}")
  local -a remaining=()
  while ((${#active[@]})); do
    if wait -n -p finished_pid "${active[@]}"; then
      status=0
    else
      status=$?
    fi
    if ((status != 0)); then
      kill "${active[@]}" 2>/dev/null || true
      wait "${active[@]}" 2>/dev/null || true
      child_pids=()
      return "${status}"
    fi
    remaining=()
    for pid in "${active[@]}"; do
      if [[ "${pid}" != "${finished_pid}" ]]; then
        remaining+=("${pid}")
      fi
    done
    active=("${remaining[@]}")
  done
  child_pids=()
}

trap cleanup_children EXIT INT TERM

"${PYTHON_BIN}" -m temporun_pipeline make-manifest \
  --shots-dir "${SHOTS_DIR}" \
  --video-root "${VIDEO_ROOT}" \
  --output "${CORPUS_ROOT}/frames.jsonl" \
  --long-shot-fps 0.25 \
  --strict-videos

"${PYTHON_BIN}" -m temporun_pipeline embed-corpus \
  --frame-manifest "${CORPUS_ROOT}/frames.jsonl" \
  --output-dir "${CORPUS_ROOT}/corpus_v2_gpu0" \
  --model "${EMBED_MODEL}" \
  --qwen-source "${QWEN_SOURCE}" \
  --device cuda:0 \
  --batch-size 8 \
  --shard-size 512 \
  --embedding-dim 4096 \
  --max-image-side 512 \
  --partition-count 2 \
  --partition-index 0 \
  --decoder-backend torchcodec \
  --prefetch &
embed_pid_0=$!

"${PYTHON_BIN}" -m temporun_pipeline embed-corpus \
  --frame-manifest "${CORPUS_ROOT}/frames.jsonl" \
  --output-dir "${CORPUS_ROOT}/corpus_v2_gpu1" \
  --model "${EMBED_MODEL}" \
  --qwen-source "${QWEN_SOURCE}" \
  --device cuda:1 \
  --batch-size 8 \
  --shard-size 512 \
  --embedding-dim 4096 \
  --max-image-side 512 \
  --partition-count 2 \
  --partition-index 1 \
  --decoder-backend torchcodec \
  --prefetch &
embed_pid_1=$!

child_pids=("${embed_pid_0}" "${embed_pid_1}")
wait_for_children

if [[ "${REUSE_RETRIEVAL_PARTITIONS}" == "true" \
      && -s "${TASK_OUTPUT_ROOT}/top3000_candidates_gpu0.jsonl" \
      && -s "${TASK_OUTPUT_ROOT}/top3000_candidates_gpu1.jsonl" ]]; then
  echo "[pipeline] Reusing existing top-3000 retrieval partitions."
else
  "${PYTHON_BIN}" -m temporun_pipeline retrieve \
    --tasks "${TASKS_PATH}" \
    --corpus-dir "${CORPUS_ROOT}/corpus_v2_gpu0" \
    --corpus-dir "${CORPUS_ROOT}/corpus_v2_gpu1" \
    --output "${TASK_OUTPUT_ROOT}/top3000_candidates_gpu0.jsonl" \
    --model "${EMBED_MODEL}" \
    --qwen-source "${QWEN_SOURCE}" \
    --device cuda:0 \
    --query-batch-size 8 \
    --top-k 3000 \
    --partition-count 2 \
    --partition-index 0 &
  retrieve_pid_0=$!

  "${PYTHON_BIN}" -m temporun_pipeline retrieve \
    --tasks "${TASKS_PATH}" \
    --corpus-dir "${CORPUS_ROOT}/corpus_v2_gpu0" \
    --corpus-dir "${CORPUS_ROOT}/corpus_v2_gpu1" \
    --output "${TASK_OUTPUT_ROOT}/top3000_candidates_gpu1.jsonl" \
    --model "${EMBED_MODEL}" \
    --qwen-source "${QWEN_SOURCE}" \
    --device cuda:1 \
    --query-batch-size 8 \
    --top-k 3000 \
    --partition-count 2 \
    --partition-index 1 &
  retrieve_pid_1=$!

  child_pids=("${retrieve_pid_0}" "${retrieve_pid_1}")
  wait_for_children
fi

"${PYTHON_BIN}" -m temporun_pipeline merge-candidates \
  --tasks "${TASKS_PATH}" \
  --input "${TASK_OUTPUT_ROOT}/top3000_candidates_gpu0.jsonl" \
  --input "${TASK_OUTPUT_ROOT}/top3000_candidates_gpu1.jsonl" \
  --output "${TASK_OUTPUT_ROOT}/top3000_candidates.jsonl"

"${PYTHON_BIN}" -m temporun_pipeline rerank \
  --tasks "${TASKS_PATH}" \
  --candidates "${TASK_OUTPUT_ROOT}/top3000_candidates.jsonl" \
  --shots-dir "${SHOTS_DIR}" \
  --video-root "${VIDEO_ROOT}" \
  --output-dir "${TASK_OUTPUT_ROOT}/rerank_top200_gpu0" \
  --model "${RERANK_MODEL}" \
  --qwen-source "${QWEN_SOURCE}" \
  --device cuda:0 \
  --candidate-limit 200 \
  --fine-top-k 30 \
  --fine-window-ms 750 \
  --fine-fps 8 \
  --long-shot-fps 1 \
  --partition-count 2 \
  --partition-index 0 \
  --decoder-backend torchcodec &
rerank_pid_0=$!

"${PYTHON_BIN}" -m temporun_pipeline rerank \
  --tasks "${TASKS_PATH}" \
  --candidates "${TASK_OUTPUT_ROOT}/top3000_candidates.jsonl" \
  --shots-dir "${SHOTS_DIR}" \
  --video-root "${VIDEO_ROOT}" \
  --output-dir "${TASK_OUTPUT_ROOT}/rerank_top200_gpu1" \
  --model "${RERANK_MODEL}" \
  --qwen-source "${QWEN_SOURCE}" \
  --device cuda:1 \
  --candidate-limit 200 \
  --fine-top-k 30 \
  --fine-window-ms 750 \
  --fine-fps 8 \
  --long-shot-fps 1 \
  --partition-count 2 \
  --partition-index 1 \
  --decoder-backend torchcodec &
rerank_pid_1=$!

child_pids=("${rerank_pid_0}" "${rerank_pid_1}")
wait_for_children

"${PYTHON_BIN}" -m temporun_pipeline merge-submissions \
  --tasks "${TASKS_PATH}" \
  --input "${TASK_OUTPUT_ROOT}/rerank_top200_gpu0/submission.json" \
  --input "${TASK_OUTPUT_ROOT}/rerank_top200_gpu1/submission.json" \
  --output-dir "${TASK_OUTPUT_ROOT}/submission_top200"

trap - EXIT INT TERM
