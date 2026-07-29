from __future__ import annotations

import gc
import heapq
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import atomic_write_jsonl, load_tasks, read_jsonl
from .models import DEFAULT_EMBED_MODEL, DEFAULT_INSTRUCTION, load_embedder
from .progress import ProgressMeter


@dataclass
class TopKUnique:
    limit: int
    _heap: list[tuple[float, int, str]] = field(default_factory=list)
    _state: dict[str, tuple[float, int, dict[str, Any]]] = field(default_factory=dict)
    _serial: int = 0

    def _prune(self) -> None:
        while self._heap:
            score, serial, uid = self._heap[0]
            current = self._state.get(uid)
            if current is not None and current[0] == score and current[1] == serial:
                break
            heapq.heappop(self._heap)

    def threshold(self) -> float:
        if len(self._state) < self.limit:
            return float("-inf")
        self._prune()
        return self._heap[0][0]

    def update(self, uid: str, score: float, payload: dict[str, Any]) -> None:
        current = self._state.get(uid)
        if current is not None:
            if score <= current[0]:
                return
        elif len(self._state) >= self.limit and score <= self.threshold():
            return

        self._serial += 1
        entry = (float(score), self._serial, payload)
        self._state[uid] = entry
        heapq.heappush(self._heap, (entry[0], entry[1], uid))

        while len(self._state) > self.limit:
            self._prune()
            old_score, old_serial, old_uid = heapq.heappop(self._heap)
            current = self._state.get(old_uid)
            if current is not None and current[:2] == (old_score, old_serial):
                del self._state[old_uid]

    def ranked(self) -> list[dict[str, Any]]:
        values = [
            {**payload, "score": float(score)}
            for score, _serial, payload in self._state.values()
        ]
        values.sort(key=lambda row: row["score"], reverse=True)
        for rank, row in enumerate(values, start=1):
            row["rank"] = rank
            row["embedding_score"] = row["score"]
        return values


def _load_shard_metadata(path: Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path))


def retrieve_top_shots(
    *,
    tasks_path: Path,
    corpus_dirs: list[Path],
    output_path: Path,
    qwen_source: Path | None = None,
    model_name: str = DEFAULT_EMBED_MODEL,
    device: str = "cuda:0",
    query_batch_size: int = 2,
    top_k: int = 3000,
    scan_batch_size: int = 2048,
    instruction: str = DEFAULT_INSTRUCTION,
    partition_count: int = 1,
    partition_index: int = 0,
    local_files_only: bool = False,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    if partition_count <= 0:
        raise ValueError("partition_count must be positive")
    if not 0 <= partition_index < partition_count:
        raise ValueError("partition_index must be in [0, partition_count)")
    all_tasks = load_tasks(tasks_path)
    tasks = [
        task
        for index, task in enumerate(all_tasks)
        if index % partition_count == partition_index
    ]
    if not corpus_dirs:
        raise ValueError("At least one corpus directory is required")
    corpus_manifests = []
    for corpus_dir in corpus_dirs:
        with (corpus_dir / "manifest.json").open("r", encoding="utf-8") as handle:
            corpus_manifest = json.load(handle)
        if not corpus_manifest.get("complete"):
            raise ValueError(f"Corpus embedding is incomplete: {corpus_dir}")
        corpus_manifests.append((corpus_dir, corpus_manifest))
    dimensions = {int(manifest["embedding_dim"]) for _, manifest in corpus_manifests}
    if len(dimensions) != 1:
        raise ValueError(f"Corpus embedding dimensions do not match: {dimensions}")
    embedding_dim = dimensions.pop()

    embedder = load_embedder(
        model_name=model_name,
        source=qwen_source,
        device=device,
        local_files_only=local_files_only,
    )
    query_parts = []
    query_progress = ProgressMeter(total=len(tasks))
    try:
        for start in range(0, len(tasks), query_batch_size):
            batch = tasks[start : start + query_batch_size]
            inputs = [
                {
                    "text": task["description"],
                    "instruction": instruction,
                }
                for task in batch
            ]
            with torch.inference_mode():
                values = embedder.process(inputs)
                values = values[:, :embedding_dim]
                values = functional.normalize(values, p=2, dim=-1)
            query_parts.append(values.detach().cpu().float())
            print(
                "[retrieve queries] "
                + query_progress.render(
                    min(start + len(batch), len(tasks)),
                    unit="queries",
                ),
                flush=True,
            )
    finally:
        del embedder
        gc.collect()
        torch.cuda.empty_cache()

    query_matrix = torch.cat(query_parts, dim=0).to(device, dtype=torch.float16)
    trackers = [TopKUnique(top_k) for _ in tasks]
    processed_frames = 0
    total_frames = sum(
        int(manifest.get("embedded_count", 0))
        for _, manifest in corpus_manifests
    )
    scan_progress = ProgressMeter(total=total_frames)

    total_shards = sum(len(manifest["shards"]) for _, manifest in corpus_manifests)
    shard_number = 0
    for corpus_dir, corpus_manifest in corpus_manifests:
        for shard in corpus_manifest["shards"]:
            shard_number += 1
            embedding_path = corpus_dir / shard["embeddings"]
            metadata_path = corpus_dir / shard["metadata"]
            embeddings = np.load(embedding_path, mmap_mode="r")
            metadata = _load_shard_metadata(metadata_path)
            if len(embeddings) != len(metadata):
                raise ValueError(f"Shard mismatch: {embedding_path} vs {metadata_path}")

            for offset in range(0, len(metadata), scan_batch_size):
                batch_meta = metadata[offset : offset + scan_batch_size]
                batch_np = np.array(
                    embeddings[offset : offset + len(batch_meta)],
                    dtype=np.float16,
                    copy=True,
                )
                corpus_batch = torch.from_numpy(batch_np).to(device)
                with torch.inference_mode():
                    score_matrix = torch.matmul(query_matrix, corpus_batch.T)
                scores = score_matrix.float().cpu().numpy()
                del corpus_batch, score_matrix

                for query_index, row_scores in enumerate(scores):
                    tracker = trackers[query_index]
                    if len(tracker._state) >= top_k:
                        viable = np.flatnonzero(row_scores > tracker.threshold())
                        order = viable[np.argsort(-row_scores[viable])]
                    else:
                        order = np.argsort(-row_scores)
                    seen_local: set[str] = set()
                    for frame_index in order:
                        score = float(row_scores[frame_index])
                        if (
                            len(tracker._state) >= top_k
                            and score <= tracker.threshold()
                        ):
                            break
                        record = batch_meta[int(frame_index)]
                        uid = str(record["shot_uid"])
                        if uid in seen_local:
                            continue
                        seen_local.add(uid)
                        payload = {
                            key: record[key]
                            for key in (
                                "video_id",
                                "video_path",
                                "shot_id",
                                "shot_start_ms",
                                "shot_end_ms",
                                "frame_ms",
                                "sample",
                            )
                            if key in record
                        }
                        tracker.update(uid, score, payload)
                processed_frames += len(batch_meta)

            print(
                f"[retrieve] shard {shard_number}/{total_shards} "
                f"{scan_progress.render(processed_frames, unit='frames')}",
                flush=True,
            )

    rows = [
        {
            "task_id": task["task_id"],
            "candidates": tracker.ranked(),
        }
        for task, tracker in zip(tasks, trackers)
    ]
    atomic_write_jsonl(output_path, rows)
    return {
        "task_count": len(tasks),
        "top_k": top_k,
        "corpus_count": len(corpus_dirs),
        "processed_frames": processed_frames,
        "partition_count": partition_count,
        "partition_index": partition_index,
        "output_path": str(output_path.resolve()),
    }
