from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from .io_utils import atomic_write_json, atomic_write_jsonl, load_tasks, read_jsonl


def merge_candidates(
    *,
    tasks_path: Path,
    input_paths: list[Path],
    output_path: Path,
) -> dict[str, Any]:
    tasks = load_tasks(tasks_path)
    by_task: dict[str, dict[str, Any]] = {}
    for input_path in input_paths:
        for row in read_jsonl(input_path):
            task_id = str(row["task_id"])
            if task_id in by_task:
                raise ValueError(f"Duplicate candidate task: {task_id}")
            by_task[task_id] = row

    expected = [str(task["task_id"]) for task in tasks]
    missing = [task_id for task_id in expected if task_id not in by_task]
    extra = sorted(set(by_task) - set(expected))
    if missing or extra:
        raise ValueError(
            f"Candidate mismatch: missing={missing[:10]} extra={extra[:10]}"
        )
    atomic_write_jsonl(output_path, (by_task[task_id] for task_id in expected))
    return {
        "task_count": len(expected),
        "output_path": str(output_path.resolve()),
    }


def merge_submissions(
    *,
    tasks_path: Path,
    input_paths: list[Path],
    output_dir: Path,
) -> dict[str, Any]:
    tasks = load_tasks(tasks_path)
    by_task: dict[str, dict[str, Any]] = {}
    for input_path in input_paths:
        with input_path.open("r", encoding="utf-8") as handle:
            submission = json.load(handle)
        for prediction in submission.get("predictions", []):
            task_id = str(prediction["task_id"])
            if task_id in by_task:
                raise ValueError(f"Duplicate task prediction: {task_id}")
            by_task[task_id] = prediction

    expected = [str(task["task_id"]) for task in tasks]
    missing = [task_id for task_id in expected if task_id not in by_task]
    extra = sorted(set(by_task) - set(expected))
    if missing or extra:
        raise ValueError(
            f"Submission mismatch: missing={missing[:10]} extra={extra[:10]}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    merged = {"predictions": [by_task[task_id] for task_id in expected]}
    json_path = output_dir / "submission.json"
    zip_path = output_dir / "submission.zip"
    atomic_write_json(json_path, merged)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(json_path, arcname="submission.json")
    return {
        "task_count": len(expected),
        "submission": str(json_path.resolve()),
        "submission_zip": str(zip_path.resolve()),
    }
