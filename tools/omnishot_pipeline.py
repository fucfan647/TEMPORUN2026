#!/usr/bin/env python3
"""Generate V3C shot manifests with OmniShotCut.

This file intentionally contains only the OmniShotCut stage. It does not sample
frames, deduplicate images, build an index, or retrieve results.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2


HF_REPO = "uva-cv-lab/OmniShotCut"
HF_FILENAME = "OmniShotCut_ckpt.pth"
DEFAULT_OVERLAP = 20


@dataclass(frozen=True)
class VideoRow:
    video_id: str
    path: Path


def list_videos(dataset_roots: Iterable[Path]) -> list[VideoRow]:
    rows: list[VideoRow] = []
    seen_ids: set[str] = set()

    for root in dataset_roots:
        root = root.expanduser().resolve()
        video_files = sorted((root / "videos").glob("*/*.mp4"))
        if not video_files:
            raise FileNotFoundError(
                f"No videos found under {root / 'videos'} (expected videos/*/*.mp4)"
            )

        prefix = root.name.lower()
        for path in video_files:
            video_id = f"{prefix}_{path.stem}"
            if video_id in seen_ids:
                raise ValueError(f"Duplicate video_id: {video_id}")
            seen_ids.add(video_id)
            rows.append(VideoRow(video_id=video_id, path=path.resolve()))

    return rows


def shard_rows(
    rows: list[VideoRow], shard_index: int, shard_count: int
) -> list[VideoRow]:
    if shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if not 0 <= shard_index < shard_count:
        raise ValueError("--shard-index must satisfy 0 <= index < shard-count")
    return rows[shard_index::shard_count]


def video_meta(path: Path) -> tuple[float, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()

    if fps <= 0:
        fps = 25.0
    if frame_count <= 0:
        raise RuntimeError(f"Invalid frame count for {path}: {frame_count}")
    return fps, frame_count


def frame_to_ms(frame: int, fps: float) -> int:
    return int(round(frame * 1000.0 / fps))


def normalize_ranges(
    ranges: Any, fps: float, frame_count: int
) -> list[dict[str, int]]:
    if hasattr(ranges, "tolist"):
        ranges = ranges.tolist()

    shots: list[dict[str, int]] = []
    for shot_id, pair in enumerate(ranges or []):
        if len(pair) < 2:
            raise ValueError(f"Invalid OmniShotCut range: {pair!r}")

        start_frame = max(0, min(int(round(float(pair[0]))), frame_count - 1))
        end_frame = max(0, min(int(round(float(pair[1]))), frame_count - 1))
        if end_frame < start_frame:
            start_frame, end_frame = end_frame, start_frame

        shots.append(
            {
                "shot_id": shot_id,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_ms": frame_to_ms(start_frame, fps),
                "end_ms": frame_to_ms(end_frame, fps),
            }
        )
    return shots


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False) + "\n")


def resolve_checkpoint(checkpoint: str) -> Path:
    from huggingface_hub import hf_hub_download

    checkpoint_path = Path(checkpoint).expanduser()
    if checkpoint_path.is_file():
        return checkpoint_path.resolve()
    return Path(
        hf_hub_download(repo_id=checkpoint, filename=HF_FILENAME)
    ).resolve()


def load_model(checkpoint_path: Path):
    import omnishotcut

    return omnishotcut.load(str(checkpoint_path))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def write_run_config(
    args: argparse.Namespace, output: Path, checkpoint_path: Path
) -> None:
    import torch

    checkpoint_source = Path(args.omni_checkpoint).expanduser()
    checkpoint_info: dict[str, Any] = {
        "source": args.omni_checkpoint,
        "resolved_path": str(checkpoint_path),
        "sha256": file_sha256(checkpoint_path),
    }
    if checkpoint_source.is_file() and checkpoint_source.parent.parent.name == "snapshots":
        checkpoint_info["huggingface_revision"] = checkpoint_source.parent.name
    elif checkpoint_path.parent.parent.name == "snapshots":
        checkpoint_info["huggingface_revision"] = checkpoint_path.parent.name

    backbone_path = (
        Path(torch.hub.get_dir()) / "checkpoints" / "resnet18-f37072fd.pth"
    )
    backbone_info: dict[str, Any] | None = None
    if backbone_path.is_file():
        backbone_info = {
            "resolved_path": str(backbone_path.resolve()),
            "sha256": file_sha256(backbone_path),
        }

    atomic_write_json(
        output / "run_config.json",
        {
            "stage": "omnishotcut_only",
            "omnishotcut_version": package_version("omnishotcut"),
            "tested_omnishotcut_commit": args.omnishotcut_commit,
            "checkpoint": checkpoint_info,
            "checkpoint_filename": HF_FILENAME,
            "backbone_checkpoint": backbone_info,
            "mode": "clean_shot",
            "overlap_frames": args.overlap if args.overlap is not None else DEFAULT_OVERLAP,
            "overlap_source": (
                "omnishotcut_default"
                if args.overlap is None
                else "explicit_cli_argument"
            ),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "opencv_version": cv2.__version__,
            "dataset_roots": [str(path.expanduser().resolve()) for path in args.dataset_root],
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
        },
    )


def run(args: argparse.Namespace) -> int:
    output = args.out.expanduser().resolve()
    shots_dir = output / "shots"
    failures_path = output / f"failed_shots_shard{args.shard_index}.jsonl"

    rows = shard_rows(
        list_videos(args.dataset_root), args.shard_index, args.shard_count
    )
    if args.dry_run_videos is not None:
        if args.dry_run_videos < 1:
            raise ValueError("--dry-run-videos must be >= 1")
        rows = rows[: args.dry_run_videos]
    if not rows:
        raise RuntimeError("No videos selected")

    print(
        f"Selected {len(rows)} video(s), shard "
        f"{args.shard_index}/{args.shard_count}",
        flush=True,
    )
    print(
        f"Loading OmniShotCut: checkpoint={args.omni_checkpoint!r}, "
        f"mode='clean_shot', overlap={args.overlap if args.overlap is not None else DEFAULT_OVERLAP}",
        flush=True,
    )
    checkpoint_path = resolve_checkpoint(args.omni_checkpoint)
    model = load_model(checkpoint_path)
    write_run_config(args, output, checkpoint_path)
    effective_overlap = args.overlap if args.overlap is not None else DEFAULT_OVERLAP

    completed = 0
    skipped = 0
    failed = 0
    started = time.monotonic()

    for index, row in enumerate(rows, start=1):
        destination = shots_dir / f"{row.video_id}.json"
        if destination.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            fps, frame_count = video_meta(row.path)
            if args.overlap is None:
                ranges = model.inference(str(row.path), mode="clean_shot")
            else:
                ranges = model.inference(
                    str(row.path), mode="clean_shot", overlap=effective_overlap
                )
            shots = normalize_ranges(ranges, fps, frame_count)
            if not shots:
                shots = normalize_ranges([[0, frame_count - 1]], fps, frame_count)

            atomic_write_json(
                destination,
                {
                    "video_id": row.video_id,
                    "video_path": str(row.path),
                    "fps": fps,
                    "frame_count": frame_count,
                    "shots": shots,
                },
            )
            completed += 1
        except Exception as error:
            failed += 1
            append_jsonl(
                failures_path,
                {
                    "video_id": row.video_id,
                    "video_path": str(row.path),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            print(
                f"FAILED {row.video_id}: {type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )

        if index % args.log_every == 0 or index == len(rows):
            elapsed = time.monotonic() - started
            print(
                f"[{index}/{len(rows)}] completed={completed} skipped={skipped} "
                f"failed={failed} elapsed={elapsed:.1f}s",
                flush=True,
            )

    print(
        f"Done: completed={completed}, skipped={skipped}, failed={failed}, "
        f"output={shots_dir}",
        flush=True,
    )
    return 1 if failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run only OmniShotCut and write V3C shot manifests."
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("shots",),
        default="shots",
        help="Optional compatibility command; only 'shots' is supported.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        action="append",
        required=True,
        help="V3C1/V3C2 root containing videos/*/*.mp4; repeat for multiple roots.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument(
        "--omni-checkpoint",
        default=HF_REPO,
        help="Local checkpoint path or Hugging Face repository ID.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=None,
        help="Overlap frames between inference windows. Omit to use OmniShotCut's internal default.",
    )
    parser.add_argument(
        "--omnishotcut-commit",
        default=os.environ.get("OMNISHOTCUT_COMMIT", "unknown"),
        help="Recorded OmniShotCut source commit for provenance.",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--dry-run-videos",
        type=int,
        default=None,
        help="Process only the first N selected videos.",
    )
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing manifests."
    )
    args = parser.parse_args()

    if args.overlap is not None and args.overlap < 0:
        parser.error("--overlap must be >= 0")
    if args.log_every < 1:
        parser.error("--log-every must be >= 1")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
