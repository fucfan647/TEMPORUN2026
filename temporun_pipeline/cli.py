from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .embed import embed_corpus
from .manifest import build_frame_manifest
from .merge import merge_candidates, merge_submissions
from .models import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_INSTRUCTION,
    DEFAULT_RERANK_MODEL,
    default_qwen_source,
)
from .rerank import rerank_and_fine_scan
from .retrieve import retrieve_top_shots


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _common_model_arguments(parser: argparse.ArgumentParser, default_model: str) -> None:
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--qwen-source", type=_path, default=default_qwen_source())
    parser.add_argument("--local-files-only", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="temporun-pipeline",
        description="Qwen3-VL-8B 4-bit video retrieval pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check runtime dependencies and GPU")
    doctor.add_argument("--qwen-source", type=_path, default=default_qwen_source())

    manifest = subparsers.add_parser(
        "make-manifest",
        help="Create the deterministic frame sampling manifest",
    )
    manifest.add_argument("--shots-dir", type=_path, required=True)
    manifest.add_argument("--video-root", type=_path, required=True)
    manifest.add_argument("--output", type=_path, required=True)
    manifest.add_argument("--long-shot-fps", type=float, default=0.25)
    manifest.add_argument("--strict-videos", action="store_true")

    embed = subparsers.add_parser(
        "embed-corpus",
        help="Embed sampled corpus frames into resumable NumPy shards",
    )
    embed.add_argument("--frame-manifest", type=_path, required=True)
    embed.add_argument("--output-dir", type=_path, required=True)
    embed.add_argument("--device", default="cuda:0")
    embed.add_argument("--batch-size", type=int, default=2)
    embed.add_argument("--shard-size", type=int, default=2048)
    embed.add_argument("--embedding-dim", type=int, default=4096)
    embed.add_argument("--max-image-side", type=int, default=512)
    embed.add_argument("--partition-count", type=int, default=1)
    embed.add_argument("--partition-index", type=int, default=0)
    embed.add_argument(
        "--decoder-backend",
        choices=("opencv", "torchcodec"),
        default="opencv",
    )
    embed.add_argument("--prefetch", action="store_true")
    _common_model_arguments(embed, DEFAULT_EMBED_MODEL)

    retrieve = subparsers.add_parser(
        "retrieve",
        help="Embed task text and retrieve top unique shots",
    )
    retrieve.add_argument("--tasks", type=_path, required=True)
    retrieve.add_argument(
        "--corpus-dir",
        dest="corpus_dirs",
        type=_path,
        action="append",
        required=True,
        help="Repeat this option to scan multiple corpus partitions",
    )
    retrieve.add_argument("--output", type=_path, required=True)
    retrieve.add_argument("--device", default="cuda:0")
    retrieve.add_argument("--query-batch-size", type=int, default=2)
    retrieve.add_argument("--top-k", type=int, default=3000)
    retrieve.add_argument("--scan-batch-size", type=int, default=2048)
    retrieve.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    retrieve.add_argument("--partition-count", type=int, default=1)
    retrieve.add_argument("--partition-index", type=int, default=0)
    _common_model_arguments(retrieve, DEFAULT_EMBED_MODEL)

    rerank = subparsers.add_parser(
        "rerank",
        help="Rerank 500 shots and fine scan the top 30 unique shots",
    )
    rerank.add_argument("--tasks", type=_path, required=True)
    rerank.add_argument("--candidates", type=_path, required=True)
    rerank.add_argument("--shots-dir", type=_path, required=True)
    rerank.add_argument("--video-root", type=_path, required=True)
    rerank.add_argument("--output-dir", type=_path, required=True)
    rerank.add_argument("--device", default="cuda:1")
    rerank.add_argument("--candidate-limit", type=int, default=500)
    rerank.add_argument("--fine-top-k", type=int, default=30)
    rerank.add_argument("--fine-window-ms", type=int, default=750)
    rerank.add_argument("--fine-fps", type=float, default=8.0)
    rerank.add_argument("--long-shot-fps", type=float, default=1.0)
    rerank.add_argument("--score-chunk-size", type=int, default=4)
    rerank.add_argument("--max-image-side", type=int, default=512)
    rerank.add_argument("--max-predictions", type=int, default=10)
    rerank.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    rerank.add_argument("--partition-count", type=int, default=1)
    rerank.add_argument("--partition-index", type=int, default=0)
    rerank.add_argument(
        "--decoder-backend",
        choices=("opencv", "torchcodec"),
        default="opencv",
    )
    _common_model_arguments(rerank, DEFAULT_RERANK_MODEL)

    merge = subparsers.add_parser(
        "merge-submissions",
        help="Merge disjoint task-partition submissions in official task order",
    )
    merge.add_argument("--tasks", type=_path, required=True)
    merge.add_argument(
        "--input",
        dest="inputs",
        type=_path,
        action="append",
        required=True,
    )
    merge.add_argument("--output-dir", type=_path, required=True)

    merge_retrieval = subparsers.add_parser(
        "merge-candidates",
        help="Merge disjoint retrieval partitions in official task order",
    )
    merge_retrieval.add_argument("--tasks", type=_path, required=True)
    merge_retrieval.add_argument(
        "--input",
        dest="inputs",
        type=_path,
        action="append",
        required=True,
    )
    merge_retrieval.add_argument("--output", type=_path, required=True)
    return parser


def doctor(qwen_source: Path) -> dict[str, Any]:
    import importlib.util

    import torch

    modules = [
        "transformers",
        "accelerate",
        "bitsandbytes",
        "qwen_vl_utils",
        "torchcodec",
        "cv2",
        "scipy",
    ]
    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpus": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "memory_gib": round(
                    torch.cuda.get_device_properties(index).total_memory / 2**30,
                    2,
                ),
            }
            for index in range(torch.cuda.device_count())
        ],
        "modules": {
            module: importlib.util.find_spec(module) is not None for module in modules
        },
        "qwen_source": str(qwen_source),
        "qwen_source_ready": (
            qwen_source / "src" / "models" / "qwen3_vl_embedding.py"
        ).exists(),
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        result = doctor(args.qwen_source)
    elif args.command == "make-manifest":
        result = build_frame_manifest(
            shots_dir=args.shots_dir,
            video_root=args.video_root,
            output_path=args.output,
            long_shot_fps=args.long_shot_fps,
            strict_videos=args.strict_videos,
        )
    elif args.command == "embed-corpus":
        result = embed_corpus(
            frame_manifest=args.frame_manifest,
            output_dir=args.output_dir,
            qwen_source=args.qwen_source,
            model_name=args.model,
            device=args.device,
            batch_size=args.batch_size,
            shard_size=args.shard_size,
            embedding_dim=args.embedding_dim,
            max_image_side=args.max_image_side,
            partition_count=args.partition_count,
            partition_index=args.partition_index,
            decoder_backend=args.decoder_backend,
            prefetch=args.prefetch,
            local_files_only=args.local_files_only,
        )
    elif args.command == "retrieve":
        result = retrieve_top_shots(
            tasks_path=args.tasks,
            corpus_dirs=args.corpus_dirs,
            output_path=args.output,
            qwen_source=args.qwen_source,
            model_name=args.model,
            device=args.device,
            query_batch_size=args.query_batch_size,
            top_k=args.top_k,
            scan_batch_size=args.scan_batch_size,
            instruction=args.instruction,
            partition_count=args.partition_count,
            partition_index=args.partition_index,
            local_files_only=args.local_files_only,
        )
    elif args.command == "rerank":
        result = rerank_and_fine_scan(
            tasks_path=args.tasks,
            candidates_path=args.candidates,
            shots_dir=args.shots_dir,
            video_root=args.video_root,
            output_dir=args.output_dir,
            qwen_source=args.qwen_source,
            model_name=args.model,
            device=args.device,
            candidate_limit=args.candidate_limit,
            fine_top_k=args.fine_top_k,
            fine_window_ms=args.fine_window_ms,
            fine_fps=args.fine_fps,
            long_shot_fps=args.long_shot_fps,
            score_chunk_size=args.score_chunk_size,
            max_image_side=args.max_image_side,
            max_predictions=args.max_predictions,
            instruction=args.instruction,
            partition_count=args.partition_count,
            partition_index=args.partition_index,
            decoder_backend=args.decoder_backend,
            local_files_only=args.local_files_only,
        )
    elif args.command == "merge-submissions":
        result = merge_submissions(
            tasks_path=args.tasks,
            input_paths=args.inputs,
            output_dir=args.output_dir,
        )
    elif args.command == "merge-candidates":
        result = merge_candidates(
            tasks_path=args.tasks,
            input_paths=args.inputs,
            output_path=args.output,
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))
