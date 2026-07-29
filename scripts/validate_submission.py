#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_tasks(path: Path) -> list[str]:
    task_ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = str(row.get("task_id", ""))
            if not task_id:
                raise ValueError(f"Missing task_id at {path}:{line_number}")
            task_ids.append(task_id)
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Input task IDs are not unique")
    return task_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a TempoRun submission.")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--shots-dir", type=Path)
    args = parser.parse_args()

    expected = read_tasks(args.tasks)
    with args.submission.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if set(payload) != {"predictions"}:
        raise ValueError("The only top-level key must be 'predictions'")
    predictions = payload["predictions"]
    if not isinstance(predictions, list):
        raise TypeError("'predictions' must be a list")

    actual = [str(row.get("task_id", "")) for row in predictions]
    if actual != expected:
        raise ValueError("Prediction task IDs or ordering do not match input tasks")

    known_videos = None
    if args.shots_dir:
        known_videos = {path.stem for path in args.shots_dir.glob("*.json")}
        if not known_videos:
            raise ValueError(f"No shot JSON files found in {args.shots_dir}")

    for prediction in predictions:
        task_id = str(prediction["task_id"])
        results = prediction.get("results")
        if not isinstance(results, list):
            raise TypeError(f"{task_id}: results must be a list")
        if len(results) > 10:
            raise ValueError(f"{task_id}: more than 10 results")

        expected_ranks = list(range(1, len(results) + 1))
        ranks = [row.get("rank") for row in results]
        if ranks != expected_ranks:
            raise ValueError(f"{task_id}: ranks must be consecutive from 1")

        videos: set[str] = set()
        for result in results:
            video_id = result.get("video_id")
            frame_ms = result.get("frame_ms")
            if not isinstance(video_id, str) or not video_id:
                raise TypeError(f"{task_id}: invalid video_id")
            if video_id in videos:
                raise ValueError(f"{task_id}: duplicate video_id {video_id}")
            videos.add(video_id)
            if known_videos is not None and video_id not in known_videos:
                raise ValueError(f"{task_id}: unknown video_id {video_id}")
            if isinstance(frame_ms, bool) or not isinstance(frame_ms, int):
                raise TypeError(f"{task_id}: frame_ms must be an integer")
            if frame_ms < 0:
                raise ValueError(f"{task_id}: frame_ms must be non-negative")

    print(
        f"VALID: {len(predictions)} tasks, "
        f"{sum(len(row['results']) for row in predictions)} results"
    )


if __name__ == "__main__":
    main()
