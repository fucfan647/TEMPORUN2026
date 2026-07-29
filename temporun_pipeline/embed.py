from __future__ import annotations

import gc
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import atomic_write_json, atomic_write_jsonl, read_jsonl, sha256_file
from .models import DEFAULT_EMBED_MODEL, load_embedder
from .progress import ProgressMeter
from .video import VideoFrameReaderCache


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".npy",
        dir=path.parent,
    )
    os.close(descriptor)
    try:
        np.save(temporary, array, allow_pickle=False)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_shard(
    output_dir: Path,
    shard_index: int,
    embeddings: list[np.ndarray],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    if not embeddings or len(embeddings) != len(records):
        raise ValueError("Embedding shard is empty or inconsistent")
    stem = f"corpus-{shard_index:05d}"
    embedding_path = output_dir / f"{stem}.npy"
    metadata_path = output_dir / f"{stem}.jsonl"
    matrix = np.stack(embeddings).astype(np.float16, copy=False)
    _atomic_save_npy(embedding_path, matrix)
    atomic_write_jsonl(metadata_path, records)
    return {
        "index": shard_index,
        "count": len(records),
        "dimension": int(matrix.shape[1]),
        "embeddings": embedding_path.name,
        "metadata": metadata_path.name,
        "first_source_index": int(records[0]["source_index"]),
        "last_source_index": int(records[-1]["source_index"]),
    }


def embed_corpus(
    *,
    frame_manifest: Path,
    output_dir: Path,
    qwen_source: Path | None = None,
    model_name: str = DEFAULT_EMBED_MODEL,
    device: str = "cuda:0",
    batch_size: int = 2,
    shard_size: int = 2048,
    embedding_dim: int = 4096,
    max_image_side: int = 512,
    partition_count: int = 1,
    partition_index: int = 0,
    decoder_backend: str = "opencv",
    prefetch: bool = False,
    local_files_only: bool = False,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    if batch_size <= 0 or shard_size <= 0:
        raise ValueError("batch_size and shard_size must be positive")
    if not 64 <= embedding_dim <= 4096:
        raise ValueError("embedding_dim must be between 64 and 4096")
    if partition_count <= 0:
        raise ValueError("partition_count must be positive")
    if not 0 <= partition_index < partition_count:
        raise ValueError("partition_index must be in [0, partition_count)")
    if decoder_backend not in {"opencv", "torchcodec"}:
        raise ValueError("decoder_backend must be 'opencv' or 'torchcodec'")

    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "manifest.json"
    source_hash = sha256_file(frame_manifest)
    with frame_manifest.open("rb") as handle:
        source_total = sum(
            chunk.count(b"\n")
            for chunk in iter(lambda: handle.read(8 << 20), b"")
        )
    partition_total = max(
        0,
        (source_total + partition_count - 1 - partition_index) // partition_count,
    )
    state: dict[str, Any]
    if state_path.exists():
        with state_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        expected = {
            "format": "temporun-qwen3-vl-corpus-v2",
            "source_sha256": source_hash,
            "model": model_name,
            "embedding_dim": embedding_dim,
            "batch_size": batch_size,
            "shard_size": shard_size,
            "max_image_side": max_image_side,
            "partition_count": partition_count,
            "partition_index": partition_index,
            "decoder_backend": decoder_backend,
            "prefetch": bool(prefetch),
            "preprocessing": "official-qwen-prefetched-v1",
        }
        actual = {key: state.get(key) for key in expected}
        if actual != expected:
            raise ValueError(
                f"Existing embedding state is incompatible: {actual}; expected {expected}"
            )
    else:
        state = {
            "format": "temporun-qwen3-vl-corpus-v2",
            "source_manifest": str(frame_manifest.resolve()),
            "source_sha256": source_hash,
            "model": model_name,
            "quantization": "bitsandbytes-nf4-double-quant",
            "embedding_dim": embedding_dim,
            "batch_size": batch_size,
            "shard_size": shard_size,
            "max_image_side": max_image_side,
            "partition_count": partition_count,
            "partition_index": partition_index,
            "decoder_backend": decoder_backend,
            "prefetch": bool(prefetch),
            "preprocessing": "official-qwen-prefetched-v1",
            "source_cursor": -1,
            "embedded_count": 0,
            "decode_failure_count": 0,
            "shards": [],
            "complete": False,
        }
        atomic_write_json(state_path, state)

    if state.get("complete"):
        return state

    source_cursor = int(state.get("source_cursor", -1))
    shard_index = len(state["shards"])
    initial_processed = min(
        int(state["embedded_count"]) + int(state["decode_failure_count"]),
        partition_total,
    )
    progress = ProgressMeter(
        total=partition_total,
        initial_completed=initial_processed,
    )
    embedder = load_embedder(
        model_name=model_name,
        source=qwen_source,
        device=device,
        local_files_only=local_files_only,
    )

    shard_embeddings: list[np.ndarray] = []
    shard_records: list[dict[str, Any]] = []
    last_scanned_index = source_cursor
    started_at = time.monotonic()

    def commit_shard() -> None:
        nonlocal shard_index, shard_embeddings, shard_records
        if not shard_records:
            return
        shard = _write_shard(
            output_dir,
            shard_index,
            shard_embeddings,
            shard_records,
        )
        state["shards"].append(shard)
        state["source_cursor"] = last_scanned_index
        state["embedded_count"] = int(state["embedded_count"]) + len(shard_records)
        atomic_write_json(state_path, state)
        print(
            f"[embed p{partition_index}/{partition_count}] "
            f"shard={shard_index:05d} count={len(shard_records)} "
            f"{progress.render(int(state['embedded_count']) + int(state['decode_failure_count']), unit='frames')} "
            f"decode_failures={state['decode_failure_count']} cursor={last_scanned_index}",
            flush=True,
        )
        shard_index += 1
        shard_embeddings = []
        shard_records = []

    def run_batch(
        records: list[dict[str, Any]],
        processed_inputs: dict[str, Any],
    ) -> None:
        if not records:
            return
        with torch.inference_mode():
            model_inputs = {
                key: value.to(embedder.model.device, non_blocking=True)
                for key, value in processed_inputs.items()
            }
            outputs = embedder.forward(model_inputs)
            values = embedder._pooling_last(
                outputs["last_hidden_state"],
                outputs["attention_mask"],
            )
            values = values[:, :embedding_dim]
            values = functional.normalize(values, p=2, dim=-1)
        matrix = values.detach().cpu().float().numpy()
        shard_embeddings.extend(matrix)
        shard_records.extend(records)
        del values, matrix, model_inputs, outputs

    def decoded_batches():
        with VideoFrameReaderCache(
            max_open=2,
            max_image_side=max_image_side,
            decoder_backend=decoder_backend,
            device=device,
        ) as reader:
            records: list[dict[str, Any]] = []
            images: list[Any] = []
            failures = 0
            batch_cursor = source_cursor
            for record in read_jsonl(frame_manifest):
                source_index = int(record["source_index"])
                if source_index <= source_cursor:
                    continue
                if source_index % partition_count != partition_index:
                    continue
                batch_cursor = source_index
                image = reader.read(Path(record["video_path"]), int(record["frame_ms"]))
                if image is None:
                    failures += 1
                else:
                    records.append(record)
                    images.append(image)
                if len(records) + failures >= batch_size:
                    yield records, images, failures, batch_cursor
                    records, images, failures = [], [], 0
            if records or failures:
                yield records, images, failures, batch_cursor

    def next_batch(iterator):
        try:
            records, images, failures, batch_cursor = next(iterator)
        except StopIteration:
            return None
        if records:
            conversations = [
                embedder.format_model_input(image=image)
                for image in images
            ]
            processed_inputs = embedder._preprocess_inputs(conversations)
            processed_inputs = {
                key: (
                    value.pin_memory()
                    if value.device.type == "cpu" and not value.is_pinned()
                    else value
                )
                for key, value in processed_inputs.items()
            }
        else:
            processed_inputs = {}
        return records, processed_inputs, failures, batch_cursor

    try:
        iterator = iter(decoded_batches())
        if prefetch:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(next_batch, iterator)
                while True:
                    batch = future.result()
                    if batch is None:
                        break
                    future = executor.submit(next_batch, iterator)
                    records, processed_inputs, failures, batch_cursor = batch
                    last_scanned_index = batch_cursor
                    state["decode_failure_count"] = (
                        int(state["decode_failure_count"]) + failures
                    )
                    run_batch(records, processed_inputs)
                    if len(shard_records) >= shard_size:
                        commit_shard()
        else:
            while True:
                batch = next_batch(iterator)
                if batch is None:
                    break
                records, processed_inputs, failures, batch_cursor = batch
                last_scanned_index = batch_cursor
                state["decode_failure_count"] = (
                    int(state["decode_failure_count"]) + failures
                )
                run_batch(records, processed_inputs)
                if len(shard_records) >= shard_size:
                    commit_shard()
        commit_shard()
    finally:
        del embedder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    state["source_cursor"] = last_scanned_index
    state["complete"] = True
    state["elapsed_seconds"] = round(time.monotonic() - started_at, 3)
    atomic_write_json(state_path, state)
    return state
