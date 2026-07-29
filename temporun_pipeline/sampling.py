from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from PIL import Image


@dataclass(frozen=True)
class FrameSample:
    frame_ms: int
    sample: str


def sample_frame_times(
    start_ms: int,
    end_ms: int,
    *,
    long_shot_fps: float = 0.25,
) -> list[FrameSample]:
    """Return deterministic sample positions for one shot.

    - duration <= 2 s: midpoint
    - duration <= 5 s: 1/3 and 2/3
    - duration > 5 s: center of each sampling interval at long_shot_fps
    """
    start_ms = max(0, int(start_ms))
    end_ms = max(start_ms + 1, int(end_ms))
    duration_ms = end_ms - start_ms

    if duration_ms <= 2_000:
        return [FrameSample(start_ms + duration_ms // 2, "mid")]

    if duration_ms <= 5_000:
        return [
            FrameSample(start_ms + duration_ms // 3, "third_1"),
            FrameSample(start_ms + (2 * duration_ms) // 3, "third_2"),
        ]

    if long_shot_fps <= 0:
        raise ValueError("long_shot_fps must be positive")
    step_ms = max(1, int(round(1_000.0 / long_shot_fps)))
    first_ms = start_ms + step_ms // 2
    values = list(range(first_ms, end_ms, step_ms))
    if not values:
        values = [start_ms + duration_ms // 2]
    return [
        FrameSample(frame_ms, f"long_{index:04d}")
        for index, frame_ms in enumerate(values)
    ]


def fine_scan_times(
    center_ms: int,
    shot_start_ms: int,
    shot_end_ms: int,
    *,
    window_ms: int = 750,
    fps: float = 8.0,
) -> list[int]:
    if window_ms < 0:
        raise ValueError("window_ms must be non-negative")
    if fps <= 0:
        raise ValueError("fps must be positive")

    shot_start_ms = max(0, int(shot_start_ms))
    shot_end_ms = max(shot_start_ms, int(shot_end_ms))
    center_ms = min(shot_end_ms, max(shot_start_ms, int(center_ms)))
    scan_start = max(shot_start_ms, center_ms - int(window_ms))
    scan_end = min(shot_end_ms, center_ms + int(window_ms))
    step_ms = max(1, int(round(1_000.0 / fps)))

    values: set[int] = {center_ms}
    offset = -int(window_ms)
    while offset <= int(window_ms):
        frame_ms = center_ms + offset
        if scan_start <= frame_ms <= scan_end:
            values.add(frame_ms)
        offset += step_ms
    return sorted(values)


def resize_keep_aspect(image: Image.Image, max_side: int) -> Image.Image:
    """Downscale to an integer size while preserving the original aspect ratio."""
    if max_side <= 0:
        return image
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = float(max_side) / float(longest)
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return image.resize(new_size, Image.Resampling.LANCZOS)


def unique_ints(values: Iterable[int]) -> list[int]:
    return sorted({int(value) for value in values})
