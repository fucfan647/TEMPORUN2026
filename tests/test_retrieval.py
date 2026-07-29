import json
from pathlib import Path

from temporun_pipeline.manifest import build_frame_manifest
from temporun_pipeline.merge import merge_candidates, merge_submissions
from temporun_pipeline.retrieve import TopKUnique
from temporun_pipeline.io_utils import read_jsonl


def test_top_k_is_unique_and_accepts_late_improvement():
    tracker = TopKUnique(2)
    tracker.update("a", 1.0, {"shot_uid": "a", "frame_ms": 10})
    tracker.update("b", 2.0, {"shot_uid": "b", "frame_ms": 20})
    tracker.update("c", 1.5, {"shot_uid": "c", "frame_ms": 30})
    tracker.update("a", 3.0, {"shot_uid": "a", "frame_ms": 40})
    tracker.update("b", 1.0, {"shot_uid": "b", "frame_ms": 50})

    ranked = tracker.ranked()
    assert [row["shot_uid"] for row in ranked] == ["a", "b"]
    assert [row["frame_ms"] for row in ranked] == [40, 20]


def test_manifest_uses_expected_sampling(tmp_path):
    shots_dir = tmp_path / "shots"
    shots_dir.mkdir()
    shot_path = shots_dir / "v3c1_00001.json"
    shot_path.write_text(
        json.dumps(
            {
                "video_id": "v3c1_00001",
                "shots": [
                    {"shot_id": 0, "start_ms": 0, "end_ms": 2_000},
                    {"shot_id": 1, "start_ms": 2_000, "end_ms": 5_000},
                    {"shot_id": 2, "start_ms": 5_000, "end_ms": 18_000},
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "frames.jsonl"
    stats = build_frame_manifest(
        shots_dir=shots_dir,
        video_root=tmp_path / "videos",
        output_path=output,
        long_shot_fps=0.25,
    )

    rows = list(read_jsonl(output))
    assert stats["frame_count"] == 6
    assert [row["frame_ms"] for row in rows] == [
        1_000,
        3_000,
        4_000,
        7_000,
        11_000,
        15_000,
    ]


def test_merge_submissions_restores_task_order(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        "\n".join(
            [
                json.dumps({"task_id": "T1", "description": "one"}),
                json.dumps({"task_id": "T2", "description": "two"}),
                json.dumps({"task_id": "T3", "description": "three"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    first = tmp_path / "first.json"
    first.write_text(
        json.dumps(
            {
                "predictions": [
                    {"task_id": "T1", "results": []},
                    {"task_id": "T3", "results": []},
                ]
            }
        ),
        encoding="utf-8",
    )
    second = tmp_path / "second.json"
    second.write_text(
        json.dumps(
            {"predictions": [{"task_id": "T2", "results": []}]}
        ),
        encoding="utf-8",
    )

    result = merge_submissions(
        tasks_path=tasks,
        input_paths=[first, second],
        output_dir=tmp_path / "merged",
    )
    with Path(result["submission"]).open("r", encoding="utf-8") as handle:
        merged = json.load(handle)
    assert [
        prediction["task_id"] for prediction in merged["predictions"]
    ] == ["T1", "T2", "T3"]


def test_merge_candidates_restores_task_order(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        "\n".join(
            json.dumps({"task_id": task_id, "description": task_id})
            for task_id in ("T1", "T2", "T3")
        )
        + "\n",
        encoding="utf-8",
    )
    first = tmp_path / "first.jsonl"
    first.write_text(
        json.dumps({"task_id": "T1", "candidates": [{"rank": 1}]})
        + "\n"
        + json.dumps({"task_id": "T3", "candidates": [{"rank": 3}]})
        + "\n",
        encoding="utf-8",
    )
    second = tmp_path / "second.jsonl"
    second.write_text(
        json.dumps({"task_id": "T2", "candidates": [{"rank": 2}]}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "merged.jsonl"
    merge_candidates(
        tasks_path=tasks,
        input_paths=[first, second],
        output_path=output,
    )
    assert [row["task_id"] for row in read_jsonl(output)] == ["T1", "T2", "T3"]
