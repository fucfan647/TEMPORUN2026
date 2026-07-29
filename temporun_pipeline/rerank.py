from __future__ import annotations

import gc
import json
import time
import zipfile
from pathlib import Path
from typing import Any

from .io_utils import (
    append_jsonl,
    atomic_write_json,
    load_tasks,
    read_jsonl,
    resolve_video_path,
    sha256_file,
)
from .models import DEFAULT_INSTRUCTION, DEFAULT_RERANK_MODEL, load_reranker
from .progress import ProgressMeter
from .sampling import fine_scan_times, sample_frame_times
from .video import VideoFrameReaderCache


class ShotMetadataCache:
    def __init__(self, shots_dir: Path):
        self.shots_dir = shots_dir
        self.cache: dict[str, dict[int, dict[str, Any]]] = {}

    def get(self, video_id: str, shot_id: int) -> dict[str, Any] | None:
        if video_id not in self.cache:
            path = self.shots_dir / f"{video_id}.json"
            if not path.exists():
                self.cache[video_id] = {}
            else:
                with path.open("r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
                self.cache[video_id] = {
                    int(shot["shot_id"]): shot
                    for shot in metadata.get("shots", [])
                }
        return self.cache[video_id].get(int(shot_id))


def _load_candidates(path: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        str(row["task_id"]): list(row.get("candidates") or row.get("results") or [])
        for row in read_jsonl(path)
    }


def _top_unique_shots(
    candidates: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    output = []
    seen: set[tuple[str, int]] = set()
    for candidate in candidates:
        key = (str(candidate["video_id"]), int(candidate.get("shot_id", -1)))
        if key in seen:
            continue
        seen.add(key)
        output.append(candidate)
        if len(output) >= limit:
            break
    return output


def rerank_and_fine_scan(
    *,
    tasks_path: Path,
    candidates_path: Path,
    shots_dir: Path,
    video_root: Path,
    output_dir: Path,
    qwen_source: Path | None = None,
    model_name: str = DEFAULT_RERANK_MODEL,
    device: str = "cuda:1",
    candidate_limit: int = 500,
    fine_top_k: int = 30,
    fine_window_ms: int = 750,
    fine_fps: float = 8.0,
    long_shot_fps: float = 1.0,
    score_chunk_size: int = 4,
    max_image_side: int = 512,
    max_predictions: int = 10,
    instruction: str = DEFAULT_INSTRUCTION,
    partition_count: int = 1,
    partition_index: int = 0,
    decoder_backend: str = "opencv",
    local_files_only: bool = False,
) -> dict[str, Any]:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "rerank_progress.jsonl"
    ranked_path = output_dir / "ranked_frames.jsonl"
    config_path = output_dir / "run_config.json"
    if partition_count <= 0:
        raise ValueError("partition_count must be positive")
    if not 0 <= partition_index < partition_count:
        raise ValueError("partition_index must be in [0, partition_count)")
    expected_config = {
        "format": "temporun-qwen3-vl-rerank-v2",
        "tasks_sha256": sha256_file(tasks_path),
        "candidates_sha256": sha256_file(candidates_path),
        "shots_dir": str(shots_dir.resolve()),
        "video_root": str(video_root.resolve()),
        "model": model_name,
        "quantization": "bitsandbytes-nf4-double-quant",
        "candidate_limit": candidate_limit,
        "fine_top_k": fine_top_k,
        "fine_window_ms": fine_window_ms,
        "fine_fps": fine_fps,
        "long_shot_fps": long_shot_fps,
        "score_chunk_size": score_chunk_size,
        "max_image_side": max_image_side,
        "max_predictions": max_predictions,
        "instruction": instruction,
        "partition_count": partition_count,
        "partition_index": partition_index,
        "decoder_backend": decoder_backend,
        "scoring": "true-batched-pairs-v1",
    }
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            actual_config = json.load(handle)
        if actual_config != expected_config:
            raise ValueError(
                "Existing rerank state has a different run configuration"
            )
    else:
        if progress_path.exists() or ranked_path.exists():
            raise ValueError(
                "Refusing to resume rerank output without run_config.json"
            )
        atomic_write_json(config_path, expected_config)
    all_tasks = load_tasks(tasks_path)
    tasks = [
        task
        for index, task in enumerate(all_tasks)
        if index % partition_count == partition_index
    ]
    candidates_by_task = _load_candidates(candidates_path)
    completed: dict[str, dict[str, Any]] = {}
    if progress_path.exists():
        for row in read_jsonl(progress_path):
            if row.get("result"):
                completed[str(row["task_id"])] = row["result"]

    reranker = load_reranker(
        model_name=model_name,
        source=qwen_source,
        device=device,
        local_files_only=local_files_only,
    )
    shot_cache = ShotMetadataCache(shots_dir)
    predictions: dict[str, dict[str, Any]] = dict(completed)
    run_started = time.monotonic()
    completed_task_count = sum(
        str(task["task_id"]) in predictions
        for task in tasks
    )
    task_progress = ProgressMeter(
        total=len(tasks),
        initial_completed=completed_task_count,
    )

    def score_items(
        query: str,
        items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        scored: list[dict[str, Any]] = []
        for offset in range(0, len(items), score_chunk_size):
            chunk = items[offset : offset + score_chunk_size]
            pairs = [
                reranker.format_mm_instruction(
                    query_text=query,
                    doc_image=item["image"],
                    instruction=instruction,
                )
                for item in chunk
            ]
            tokenized_inputs = reranker.tokenize(pairs)
            tokenized_inputs = {
                key: value.to(reranker.model.device, non_blocking=True)
                for key, value in tokenized_inputs.items()
            }
            scores = reranker.compute_scores(tokenized_inputs)
            for item, score in zip(chunk, scores):
                row = {key: value for key, value in item.items() if key != "image"}
                row["reranker_score"] = float(score)
                scored.append(row)
        return scored

    try:
        with VideoFrameReaderCache(
            max_open=2,
            max_image_side=max_image_side,
            decoder_backend=decoder_backend,
            device=device,
        ) as reader:
            for task_index, task in enumerate(tasks, start=1):
                task_id = str(task["task_id"])
                if task_id in predictions:
                    print(f"[rerank] {task_id} resume skip", flush=True)
                    continue
                task_started = time.monotonic()
                query = str(task["description"])
                raw_candidates = candidates_by_task.get(task_id)
                if raw_candidates is None:
                    raise KeyError(f"No candidates for task {task_id}")
                candidates = _top_unique_shots(raw_candidates, candidate_limit)
                decode_candidates = sorted(
                    candidates,
                    key=lambda row: (
                        str(row["video_id"]),
                        int(row.get("shot_start_ms", row.get("frame_ms", 0))),
                        int(row.get("shot_id", -1)),
                    ),
                )

                coarse_pending: list[dict[str, Any]] = []
                coarse_scored: list[dict[str, Any]] = []
                for candidate in decode_candidates:
                    video_id = str(candidate["video_id"])
                    shot_id = int(candidate.get("shot_id", -1))
                    shot = shot_cache.get(video_id, shot_id)
                    start_ms = int(
                        candidate.get(
                            "shot_start_ms",
                            shot["start_ms"] if shot else candidate.get("frame_ms", 0),
                        )
                    )
                    end_ms = int(
                        candidate.get(
                            "shot_end_ms",
                            shot["end_ms"] if shot else candidate.get("frame_ms", 0),
                        )
                    )
                    video_path = resolve_video_path(
                        video_root,
                        video_id,
                        candidate.get("video_path"),
                    )
                    if not video_path.exists():
                        continue
                    for sample in sample_frame_times(
                        start_ms,
                        end_ms,
                        long_shot_fps=long_shot_fps,
                    ):
                        image = reader.read(video_path, sample.frame_ms)
                        if image is None:
                            continue
                        coarse_pending.append(
                            {
                                "video_id": video_id,
                                "video_path": str(video_path),
                                "shot_uid": f"{video_id}:{shot_id}",
                                "shot_id": shot_id,
                                "shot_start_ms": start_ms,
                                "shot_end_ms": end_ms,
                                "frame_ms": sample.frame_ms,
                                "sample": sample.sample,
                                "embedding_score": float(
                                    candidate.get(
                                        "embedding_score",
                                        candidate.get("score", 0.0),
                                    )
                                ),
                                "scan_stage": "coarse",
                                "image": image,
                            }
                        )
                        if len(coarse_pending) >= score_chunk_size:
                            coarse_scored.extend(score_items(query, coarse_pending))
                            coarse_pending.clear()
                if coarse_pending:
                    coarse_scored.extend(score_items(query, coarse_pending))
                    coarse_pending.clear()
                coarse_scored.sort(
                    key=lambda row: (
                        row["reranker_score"],
                        row["embedding_score"],
                    ),
                    reverse=True,
                )

                best_by_shot: dict[str, dict[str, Any]] = {}
                for row in coarse_scored:
                    best_by_shot.setdefault(str(row["shot_uid"]), row)
                fine_centers = sorted(
                    best_by_shot.values(),
                    key=lambda row: (
                        row["reranker_score"],
                        row["embedding_score"],
                    ),
                    reverse=True,
                )[:fine_top_k]

                seen_frames = {
                    (str(row["shot_uid"]), int(row["frame_ms"]))
                    for row in coarse_scored
                }
                fine_items: list[dict[str, Any]] = []
                for center in sorted(
                    fine_centers,
                    key=lambda row: (
                        str(row["video_id"]),
                        int(row["frame_ms"]),
                    ),
                ):
                    for frame_ms in fine_scan_times(
                        int(center["frame_ms"]),
                        int(center["shot_start_ms"]),
                        int(center["shot_end_ms"]),
                        window_ms=fine_window_ms,
                        fps=fine_fps,
                    ):
                        key = (str(center["shot_uid"]), frame_ms)
                        if key in seen_frames:
                            continue
                        seen_frames.add(key)
                        image = reader.read(Path(center["video_path"]), frame_ms)
                        if image is None:
                            continue
                        fine_items.append(
                            {
                                **{
                                    key: center[key]
                                    for key in (
                                        "video_id",
                                        "video_path",
                                        "shot_uid",
                                        "shot_id",
                                        "shot_start_ms",
                                        "shot_end_ms",
                                        "embedding_score",
                                    )
                                },
                                "frame_ms": frame_ms,
                                "fine_center_ms": int(center["frame_ms"]),
                                "sample": "fine",
                                "scan_stage": "fine",
                                "image": image,
                            }
                        )

                fine_scored = score_items(query, fine_items)
                all_scored = coarse_scored + fine_scored
                all_scored.sort(
                    key=lambda row: (
                        row["reranker_score"],
                        1 if row["scan_stage"] == "fine" else 0,
                        row["embedding_score"],
                    ),
                    reverse=True,
                )

                selected = []
                seen_videos: set[str] = set()
                for row in all_scored:
                    video_id = str(row["video_id"])
                    if video_id in seen_videos:
                        continue
                    seen_videos.add(video_id)
                    selected.append(
                        {
                            "rank": len(selected) + 1,
                            "video_id": video_id,
                            "frame_ms": int(row["frame_ms"]),
                        }
                    )
                    if len(selected) >= max_predictions:
                        break

                if len(selected) < max_predictions:
                    for candidate in candidates:
                        video_id = str(candidate["video_id"])
                        if video_id in seen_videos:
                            continue
                        seen_videos.add(video_id)
                        selected.append(
                            {
                                "rank": len(selected) + 1,
                                "video_id": video_id,
                                "frame_ms": int(candidate.get("frame_ms", 0)),
                            }
                        )
                        if len(selected) >= max_predictions:
                            break

                result = {"task_id": task_id, "results": selected}
                predictions[task_id] = result
                completed_task_count += 1
                ranked_rows = [
                    {
                        **{
                            key: row[key]
                            for key in (
                                "video_id",
                                "shot_id",
                                "frame_ms",
                                "reranker_score",
                                "embedding_score",
                                "scan_stage",
                            )
                        },
                        "rank": rank,
                    }
                    for rank, row in enumerate(all_scored[:500], start=1)
                ]
                append_jsonl(
                    ranked_path,
                    {"task_id": task_id, "ranked_frames": ranked_rows},
                )
                append_jsonl(
                    progress_path,
                    {
                        "task_id": task_id,
                        "task_index": task_index,
                        "task_total": len(tasks),
                        "candidate_shots": len(candidates),
                        "coarse_frames": len(coarse_scored),
                        "fine_centers": len(fine_centers),
                        "fine_frames": len(fine_scored),
                        "seconds": round(time.monotonic() - task_started, 3),
                        "result": result,
                    },
                )
                del fine_items, coarse_scored, fine_scored, all_scored
                print(
                    f"[rerank p{partition_index}/{partition_count}] {task_id} "
                    f"{task_progress.render(completed_task_count, unit='tasks')} "
                    f"shots={len(candidates)} fine_centers={len(fine_centers)}",
                    flush=True,
                )
    finally:
        del reranker
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    missing = [task["task_id"] for task in tasks if task["task_id"] not in predictions]
    if missing:
        raise RuntimeError(f"Missing task predictions: {missing[:10]}")
    submission = {
        "predictions": [predictions[str(task["task_id"])] for task in tasks]
    }
    submission_path = output_dir / "submission.json"
    submission_zip = output_dir / "submission.zip"
    atomic_write_json(submission_path, submission)
    with zipfile.ZipFile(
        submission_zip,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.write(submission_path, arcname="submission.json")
    return {
        "task_count": len(tasks),
        "elapsed_seconds": round(time.monotonic() - run_started, 3),
        "submission": str(submission_path.resolve()),
        "submission_zip": str(submission_zip.resolve()),
    }
