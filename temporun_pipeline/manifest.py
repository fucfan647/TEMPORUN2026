from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io_utils import atomic_write_jsonl, resolve_video_path
from .sampling import sample_frame_times


def build_frame_manifest(
    *,
    shots_dir: Path,
    video_root: Path,
    output_path: Path,
    long_shot_fps: float = 0.25,
    strict_videos: bool = False,
) -> dict[str, int | float | str]:
    shot_files = sorted(shots_dir.glob("*.json"))
    if not shot_files:
        raise FileNotFoundError(f"No shot JSON files found in {shots_dir}")

    stats: dict[str, int] = {
        "video_count": 0,
        "shot_count": 0,
        "frame_count": 0,
        "missing_video_count": 0,
    }

    def rows():
        source_index = 0
        for shot_path in shot_files:
            with shot_path.open("r", encoding="utf-8") as handle:
                metadata: dict[str, Any] = json.load(handle)
            video_id = str(metadata.get("video_id") or shot_path.stem)
            video_path = resolve_video_path(
                video_root,
                video_id,
                metadata.get("video_path"),
            )
            if not video_path.exists():
                stats["missing_video_count"] += 1
                if strict_videos:
                    raise FileNotFoundError(
                        f"Video for {video_id} not found; resolved path: {video_path}"
                    )

            stats["video_count"] += 1
            for shot in metadata.get("shots", []):
                shot_id = int(shot["shot_id"])
                start_ms = int(shot["start_ms"])
                end_ms = int(shot["end_ms"])
                stats["shot_count"] += 1
                for sample in sample_frame_times(
                    start_ms,
                    end_ms,
                    long_shot_fps=long_shot_fps,
                ):
                    yield {
                        "source_index": source_index,
                        "shot_uid": f"{video_id}:{shot_id}",
                        "video_id": video_id,
                        "video_path": str(video_path),
                        "shot_id": shot_id,
                        "shot_start_ms": start_ms,
                        "shot_end_ms": end_ms,
                        "frame_ms": sample.frame_ms,
                        "sample": sample.sample,
                    }
                    source_index += 1
                    stats["frame_count"] += 1

    atomic_write_jsonl(output_path, rows())
    return {
        **stats,
        "long_shot_fps": float(long_shot_fps),
        "output_path": str(output_path.resolve()),
    }
