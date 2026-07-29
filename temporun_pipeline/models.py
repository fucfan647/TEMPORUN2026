from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


DEFAULT_EMBED_MODEL = "Qwen/Qwen3-VL-Embedding-8B"
DEFAULT_RERANK_MODEL = "Qwen/Qwen3-VL-Reranker-8B"
DEFAULT_INSTRUCTION = "Retrieve video frames relevant to the user's query."
PIPELINE_SEED = 2026


def default_qwen_source() -> Path:
    configured = os.environ.get("QWEN3_VL_SOURCE")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[1] / "vendor" / "Qwen3-VL-Embedding"


def activate_qwen_source(source: Path | None = None) -> Path:
    source = (source or default_qwen_source()).resolve()
    embedding_module = source / "src" / "models" / "qwen3_vl_embedding.py"
    reranker_module = source / "src" / "models" / "qwen3_vl_reranker.py"
    if not embedding_module.exists() or not reranker_module.exists():
        raise FileNotFoundError(
            f"Official Qwen source is missing at {source}. Run scripts/setup_env.sh first."
        )
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return source


def _torch_device(device: str):
    import torch

    parsed = torch.device(device)
    if parsed.type != "cuda":
        raise ValueError("The 4-bit 8B configuration requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if parsed.index is None:
        parsed = torch.device("cuda:0")
    torch.cuda.set_device(parsed)
    return parsed


def configure_reproducible_inference() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.manual_seed(PIPELINE_SEED)
    torch.cuda.manual_seed_all(PIPELINE_SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False


def _quantization_config():
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_embedder(
    *,
    model_name: str = DEFAULT_EMBED_MODEL,
    source: Path | None = None,
    device: str = "cuda:0",
    max_length: int = 8192,
    local_files_only: bool = False,
):
    import torch

    configure_reproducible_inference()
    activate_qwen_source(source)
    from src.models.qwen3_vl_embedding import (
        MAX_FRAMES,
        MAX_PIXELS,
        MAX_TOTAL_PIXELS,
        MIN_PIXELS,
        Qwen3VLEmbedder,
        Qwen3VLForEmbedding,
        Qwen3VLProcessor,
    )

    target = _torch_device(device)
    embedder = object.__new__(Qwen3VLEmbedder)
    embedder.max_length = int(max_length)
    embedder.min_pixels = MIN_PIXELS
    embedder.max_pixels = MAX_PIXELS
    embedder.total_pixels = MAX_TOTAL_PIXELS
    embedder.fps = 1.0
    embedder.max_frames = MAX_FRAMES
    embedder.default_instruction = "Represent the user's input."
    embedder.model = Qwen3VLForEmbedding.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=_quantization_config(),
        device_map={"": str(target)},
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
    )
    embedder.model.config.use_cache = False
    embedder.model.config.text_config.use_cache = False
    embedder.processor = Qwen3VLProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
        local_files_only=local_files_only,
    )
    embedder.model.eval()
    return embedder


def _patch_mm_token_type_ids(reranker: Any) -> None:
    import torch

    original_tokenize = reranker.tokenize

    def fixed_tokenize(pairs, **kwargs):
        inputs = original_tokenize(pairs, **kwargs)
        if "mm_token_type_ids" not in inputs:
            return inputs
        target_len = int(inputs["input_ids"].shape[1])
        value = inputs["mm_token_type_ids"]
        if hasattr(value, "detach"):
            rows = value.detach().cpu().tolist()
            if value.ndim == 1:
                rows = [rows]
        else:
            rows = [
                row.detach().cpu().tolist() if hasattr(row, "detach") else list(row)
                for row in value
            ]
        fixed_rows = []
        for row in rows:
            row = row[-target_len:]
            fixed_rows.append(([0] * (target_len - len(row))) + row)
        inputs["mm_token_type_ids"] = torch.tensor(fixed_rows, dtype=torch.long)
        return inputs

    reranker.tokenize = fixed_tokenize


def load_reranker(
    *,
    model_name: str = DEFAULT_RERANK_MODEL,
    source: Path | None = None,
    device: str = "cuda:1",
    max_length: int = 10_240,
    local_files_only: bool = False,
):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    configure_reproducible_inference()
    activate_qwen_source(source)
    from src.models.qwen3_vl_reranker import (
        MAX_FRAMES,
        MAX_PIXELS,
        MAX_TOTAL_PIXELS,
        MIN_PIXELS,
        Qwen3VLReranker,
    )

    target = _torch_device(device)
    reranker = object.__new__(Qwen3VLReranker)
    reranker.device = target
    reranker.max_length = int(max_length)
    reranker.min_pixels = MIN_PIXELS
    reranker.max_pixels = MAX_PIXELS
    reranker.total_pixels = MAX_TOTAL_PIXELS
    reranker.fps = 1.0
    reranker.max_frames = MAX_FRAMES
    reranker.default_instruction = DEFAULT_INSTRUCTION

    language_model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=_quantization_config(),
        device_map={"": str(target)},
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
    )
    reranker.model = language_model.model
    reranker.model.config.use_cache = False
    reranker.model.config.text_config.use_cache = False
    reranker.processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
        local_files_only=local_files_only,
    )
    reranker.model.eval()
    token_yes = reranker.processor.tokenizer.get_vocab()["yes"]
    token_no = reranker.processor.tokenizer.get_vocab()["no"]
    reranker.score_linear = reranker.get_binary_linear(
        language_model,
        token_yes,
        token_no,
    )
    reranker.score_linear.eval()
    reranker.score_linear.to(target).to(reranker.model.dtype)
    _patch_mm_token_type_ids(reranker)
    return reranker
