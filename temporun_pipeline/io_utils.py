from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def video_id_parts(video_id: str) -> tuple[str, str] | None:
    match = re.fullmatch(r"v3c([12])_(\d+)", video_id.lower())
    if not match:
        return None
    return f"V3C{match.group(1)}", match.group(2)


def resolve_video_path(
    video_root: Path,
    video_id: str,
    hinted_path: str | None = None,
) -> Path:
    candidates: list[Path] = []
    if hinted_path:
        candidates.append(Path(hinted_path))

    parts = video_id_parts(video_id)
    if parts:
        collection, short_id = parts
        candidates.extend(
            [
                video_root / collection / "videos" / short_id / f"{short_id}.mp4",
                video_root / "videos" / short_id / f"{short_id}.mp4",
                video_root / short_id / f"{short_id}.mp4",
                video_root / collection / f"{short_id}.mp4",
            ]
        )
    candidates.extend(
        [
            video_root / f"{video_id}.mp4",
            video_root / video_id / f"{video_id}.mp4",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[1 if hinted_path and len(candidates) > 1 else 0]


def load_tasks(path: Path) -> list[dict[str, Any]]:
    tasks = list(read_jsonl(path))
    seen: set[str] = set()
    for task in tasks:
        task_id = str(task.get("task_id", ""))
        if not task_id:
            raise ValueError(f"Task without task_id in {path}")
        if task_id in seen:
            raise ValueError(f"Duplicate task_id {task_id!r}")
        if not task.get("description"):
            raise ValueError(f"Task {task_id!r} has no description")
        seen.add(task_id)
    return tasks
