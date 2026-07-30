# TempoRun 2026 — Qwen3-VL-8B video retrieval

This method combines shot-aware temporal sampling, Qwen3-VL multimodal
embedding retrieval, Qwen3-VL cross-modal reranking, and local temporal
refinement to retrieve a single relevant frame from each selected video.

Before submitting the repository URL, fill the three `TODO` fields below and
create the official Git commit or release tag.

## Submission information

| Field | Value |
|---|---|
| Team | `messiuuu` |
| Method | `OmniShot-Qwen3VL Retrieval` |
| Repository | `https://github.com/fucfan647/TEMPORUN2026` |
| Official commit/tag | `v1.0.0` |
| Environment method | Direct installation with `uv` |
| Tested hardware | 2 × NVIDIA GeForce RTX 4070, 12 GiB each |
| Main command | `./run_from_raw.sh` |
| Contact | `TODO: name and email` |

## 1. Method

The method combines shot-aware temporal sampling with a two-stage Qwen3-VL
retrieval framework. Videos are segmented by OmniShotCut, and representative
frames are encoded together with natural-language queries in a shared
embedding space to retrieve relevant candidates. A multimodal reranker then
evaluates the relevance between each query and candidate frame more precisely,
followed by local temporal refinement around the strongest matches. The
highest-scoring frame from each selected video is used to produce the final
ranked results.

## 2. Repository structure

```text
.
├── .gitignore
├── README.md
├── SUBMISSION_CHECKLIST.md
├── pyproject.toml
├── requirements.txt
├── requirements-omnishot.txt
├── run_from_raw.sh
├── run_pipeline.sh
├── docs/
│   └── OMNISHOTCUT_REPRODUCIBILITY.md
├── models/
│   └── README.md
├── scripts/
│   ├── download_models.sh
│   ├── download_omnishot_models.sh
│   ├── setup_env.sh
│   ├── setup_omnishot_env.sh
│   └── validate_submission.py
├── temporun_pipeline/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── embed.py
│   ├── io_utils.py
│   ├── manifest.py
│   ├── merge.py
│   ├── models.py
│   ├── progress.py
│   ├── rerank.py
│   ├── retrieve.py
│   ├── sampling.py
│   └── video.py
├── tests/
│   ├── test_progress.py
│   ├── test_retrieval.py
│   └── test_sampling.py
└── tools/
    └── omnishot_pipeline.py
```

During setup and execution, the scripts create `.venv/`, `.venv-omnishot/`,
`vendor/`, and `artifacts/`, and populate `models/` with the required
checkpoints. These generated contents are excluded from Git; only
`models/README.md` is tracked. Competition videos, task files, embeddings,
prediction artifacts, model weights, and caches are also excluded.

## 3. Tested hardware and software

The Qwen retrieval and reranking stages were run successfully with:

- Ubuntu 24.04.4 LTS;
- Python 3.12.13;
- 2 × NVIDIA GeForce RTX 4070, 12 GiB VRAM each;
- NVIDIA driver 580.142;
- system CUDA 13.0.3;
- PyTorch 2.12.0+cu130;
- Torchvision 0.27.0+cu130;
- TorchCodec 0.12.0+cu130;
- Transformers 5.14.1;
- BitsAndBytes 0.50.0;
- FFmpeg 6.1.1;
- Bash 5.1 or newer;
- 94 GiB system RAM available during the tested run.

The official launcher uses two CUDA devices concurrently and was not tested on
one GPU. The two Qwen model snapshots require 31.50 GiB in total. Reserve at
least 50 GiB beyond the V3C dataset for weights, embeddings, logs, and results.

## 4. Environment setup

This section covers only the software environment: Python, PyTorch, CUDA
runtime, Python packages, and the pinned external source code needed to import
the models. It does not download model checkpoints.

### 4.1 Installation — tested CUDA 13 path

On Ubuntu 22.04 or 24.04, install the required system tools and `uv`:

```bash
sudo apt-get update
sudo apt-get install -y git curl ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

cd /path/to/temporun_pipeline_submission

uv venv --python 3.12 .venv
export PIPELINE_PYTHON="$PWD/.venv/bin/python"

uv pip install --python "$PIPELINE_PYTHON" \
  torch==2.12.0 torchvision==0.27.0 \
  --index-url https://download.pytorch.org/whl/cu130
uv pip install --python "$PIPELINE_PYTHON" \
  torchcodec==0.12.0 \
  --index-url https://download.pytorch.org/whl/cu130

chmod +x run_pipeline.sh run_from_raw.sh scripts/*.sh
./scripts/setup_env.sh
```

`setup_env.sh` installs this package and checks out the
[official Qwen helper source](https://github.com/QwenLM/Qwen3-VL-Embedding)
at commit:

```text
393e2978d27852b0d0230d6994f37f9c15bed73c
```

For a CUDA 12 host, use compatible `cu126` PyTorch and TorchCodec wheels in the
two commands above. Do not mix TorchCodec 0.12 with PyTorch older than 2.11;
see the official
[TorchCodec compatibility table](https://github.com/meta-pytorch/torchcodec#compatibility).

Verify the main environment:

```bash
"$PIPELINE_PYTHON" -m temporun_pipeline doctor
```

### 4.2 OmniShotCut environment

The main command uses a separate environment for the OmniShotCut stage:

```bash
./scripts/setup_omnishot_env.sh
```

The script creates `.venv-omnishot` and installs the environment required by
the shot-segmentation stage.

## 5. Checkpoints and resources

This section corresponds to the checkpoint/resource requirements in
`submit_tutorial.md` section 10. It covers download URLs, revisions, sizes,
checksums, target directories, and the resources used by the method.

### 5.1 Qwen models used for the official retrieval run

Run:

```bash
export PIPELINE_PYTHON="${PIPELINE_PYTHON:-$PWD/.venv/bin/python}"
./scripts/download_models.sh
```

The target directory defaults to `models/` and can be changed with
`MODEL_ROOT`. The script downloads tokenizer/config files from pinned
Hugging Face revisions, downloads all weight shards, and verifies every shard
by byte size and SHA-256.

| Resource | Revision | Size |
|---|---|---:|
| [`Qwen/Qwen3-VL-Embedding-8B`](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) | `2c4565515e0f265c6511776e7193b22c0968ddc7` | 16,289,679,624 bytes / 15.17 GiB |
| [`Qwen/Qwen3-VL-Reranker-8B`](https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B) | `b212dc8c91a8164aef1ea2de9c1a867611e75c04` | 17,534,339,488 bytes / 16.33 GiB |

The individual weight hashes are recorded in `scripts/download_models.sh` and
`models/README.md`. No extraction step is required because the downloaded files
are model snapshot files, not archives.

### 5.2 OmniShotCut resources

Run after `setup_omnishot_env.sh`:

```bash
./scripts/download_omnishot_models.sh
```

This downloads and verifies:

| Resource | Revision | Size | SHA-256 |
|---|---|---:|---|
| [`uva-cv-lab/OmniShotCut/OmniShotCut_ckpt.pth`](https://huggingface.co/uva-cv-lab/OmniShotCut) | `7f646c4ff4bb843e18c013481fb5d9ed2b068c6b` | 164,149,963 bytes| `5948ea78e00626c0e6c5e742e64873ef872cf4a5071d2a0841aed51c3e686cfa` |
| [`resnet18-f37072fd.pth`](https://download.pytorch.org/models/resnet18-f37072fd.pth) | PyTorch official weight | 46,830,571 bytes| `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |

The targets default to `models/OmniShotCut/` and
`models/torch/hub/checkpoints/`.

The OmniShotCut source and inference settings used by the pipeline are:

- package version: `0.1.0`;
- source repository:
  `https://github.com/UVA-Computer-Vision-Lab/OmniShotCut`;
- Git commit: `3331cd3163f7f17cd6d7c8fc12ffde22894ace01`;
- inference mode: `clean_shot`;
- overlap: internal default of 20 frames.

The exact tested environment is:

- base image:
  `cnstark/pytorch:2.3.1-py3.10.15-cuda12.1.0-devel-ubuntu22.04`;
- image digest:
  `sha256:6386285f288cea3062933bde57d3db49550d86a78e1d7fb7c82efdcce3abe6a7`;
- Python 3.10.15;
- PyTorch 2.3.1+cu121 and Torchvision 0.18.1+cu121;
- OpenCV/OpenCV-headless 5.0.0.93;
- Decord 0.6.0, NumPy 2.1.2 and Pillow 10.4.0;
- FFmpeg 4.4.2.

See `docs/OMNISHOTCUT_REPRODUCIBILITY.md` for complete provenance.

## 6. Input data

Expected V3C layout:

```text
dataset/
├── V3C1/
│   └── videos/
│       └── 00001/
│           └── 00001.mp4
├── V3C2/
│   └── videos/
│       └── ...
├── public_round_tasks.jsonl
└── private_round_tasks.jsonl
```

Each non-empty task JSONL row must contain at least:

```json
{"task_id": "task_001", "description": "the natural-language query"}
```

The OmniShotCut stage automatically creates the following intermediate
manifests:

```text
omnishot_output/
└── shots/
    ├── v3c1_00001.json
    └── v3c2_....json
```

Each shot file contains `video_id`, `fps`, `frame_count`, and a `shots` array.
Every shot has integer `shot_id`, `start_frame`, `end_frame`, `start_ms`, and
`end_ms`. An absolute `video_path` stored in a manifest is only a hint; if it is
not valid on the new machine, the pipeline resolves the clip from `VIDEO_ROOT`
and `video_id`.

These manifests are generated by `run_from_raw.sh`; they are not a separate
input that the organizer needs to prepare. All external paths are supplied
using environment variables or CLI arguments. No competition data path is
fixed in the Python source.

## 7. Run the scripts

The repository-level scripts have the following roles:

| Script | Function |
|---|---|
| `scripts/setup_env.sh` | Install the main package and pinned Qwen helper source |
| `scripts/download_models.sh` | Download and verify both Qwen3-VL snapshots |
| `scripts/setup_omnishot_env.sh` | Create the pinned OmniShotCut Python environment |
| `scripts/download_omnishot_models.sh` | Download and verify OmniShotCut and ResNet18 weights |
| `tools/omnishot_pipeline.py` | Generate/resume per-video shot manifests |
| `run_pipeline.sh` | Internal launcher called by `run_from_raw.sh` |
| `run_from_raw.sh` | Generate shots, then run the full retrieval pipeline |
| `scripts/validate_submission.py` | Validate final JSON format and task coverage |

### 7.1 Generate shot manifests only

This is an optional diagnostic command for running the OmniShotCut stage by
itself. Do not run it separately before the main command:
`run_from_raw.sh` invokes the same stage automatically.

```bash
export OMNISHOT_PYTHON="$PWD/.venv-omnishot/bin/python"
export TORCH_HOME="$PWD/models/torch"

"$OMNISHOT_PYTHON" tools/omnishot_pipeline.py shots \
  --dataset-root /absolute/dataset/V3C1 \
  --dataset-root /absolute/dataset/V3C2 \
  --out /absolute/work/omnishot_output \
  --omni-checkpoint "$PWD/models/OmniShotCut/OmniShotCut_ckpt.pth" \
  --omnishotcut-commit 3331cd3163f7f17cd6d7c8fc12ffde22894ace01
```

Do not pass `--overlap` for the official setting. This deliberately uses the
OmniShotCut internal default of 20 frames. Existing valid files are skipped;
use `--overwrite` only when intentionally regenerating them.

### 7.2 Run the complete pipeline

After both environments and all models are prepared:

```bash
export PIPELINE_PYTHON="$PWD/.venv/bin/python"
export OMNISHOT_PYTHON="$PWD/.venv-omnishot/bin/python"
export VIDEO_ROOT=/absolute/dataset
export TASKS_PATH=/absolute/dataset/private_round_tasks.jsonl
export ARTIFACT_ROOT=/absolute/work/temporun_artifacts

./run_from_raw.sh
```

`run_from_raw.sh` generates/resumes shot manifests under
`$SHOT_OUTPUT/shots` (default: `$ARTIFACT_ROOT/omnishot_output/shots`) and then invokes the official two-GPU
retrieval launcher. It runs shot segmentation, frame sampling, corpus
embedding, retrieval, reranking, temporal refinement, and submission merging
without interactive input.

Optional path variables:

- `V3C1_ROOT` and `V3C2_ROOT`: override the two dataset collection roots;
- `SHOT_OUTPUT`: override the generated OmniShotCut output directory;
- `MODEL_ROOT`: defaults to `./models`;
- `CORPUS_ROOT`: defaults to `ARTIFACT_ROOT`;
- `TASK_OUTPUT_ROOT`: defaults to `ARTIFACT_ROOT`;
- `TORCH_HOME`: overrides the ResNet18 cache location;
- `EMBED_MODEL`, `RERANK_MODEL`, and `QWEN_SOURCE`: override individual
  resource locations;
- `REUSE_RETRIEVAL_PARTITIONS=true`: reuse two existing non-empty top-3000
  partition files.

### 7.3 Individual pipeline stages

```bash
"$PIPELINE_PYTHON" -m temporun_pipeline make-manifest --help
"$PIPELINE_PYTHON" -m temporun_pipeline embed-corpus --help
"$PIPELINE_PYTHON" -m temporun_pipeline retrieve --help
"$PIPELINE_PYTHON" -m temporun_pipeline merge-candidates --help
"$PIPELINE_PYTHON" -m temporun_pipeline rerank --help
"$PIPELINE_PYTHON" -m temporun_pipeline merge-submissions --help
```

The required official order is:

```text
OmniShotCut
  → frame manifest
  → two corpus embedding partitions
  → two retrieval task partitions
  → merged top-3000 candidates
  → two reranking task partitions
  → merged submission
```

## 8. Official default parameters

The values below are those passed by `run_pipeline.sh`; they are authoritative
for the submitted method even where a standalone CLI help default differs.

| Stage | Parameter | Official value |
|---|---|---:|
| OmniShotCut | mode | `clean_shot` |
| OmniShotCut | overlap | internal default, 20 frames |
| Corpus sampling, shots `>5 s` | FPS | `0.25` |
| Embedding | maximum image side | `512` |
| Embedding | batch size per GPU | `8` |
| Embedding | shard size | `512` |
| Embedding | dimension | `4096` |
| Retrieval | top unique shots | `3000` |
| Retrieval | query batch per GPU | `8` |
| Reranker | candidate shots | `200` |
| Reranker coarse sampling, shots `>5 s` | FPS | `1.0` |
| Reranker | score chunk size | `4` |
| Fine scan | top shots | `30` |
| Fine scan | window | `±750 ms` |
| Fine scan | FPS | `8.0` |
| Output | maximum predictions per task | `10` |
| Main model loading | quantization | NF4 4-bit, double quant, FP16 compute |
| Decoder | backend | TorchCodec/NVDEC |
| Task/GPU partition count | count | `2` |
| Random seed | seed | `2026` |

## 9. Output

The final files are:

```text
$TASK_OUTPUT_ROOT/submission_top200/submission.json
$TASK_OUTPUT_ROOT/submission_top200/submission.zip
```

The UTF-8 JSON format is:

```json
{
  "predictions": [
    {
      "task_id": "task_001",
      "results": [
        {"rank": 1, "video_id": "v3c1_00001", "frame_ms": 12345}
      ]
    }
  ]
}
```

The merge step restores the exact task order from the input JSONL and rejects
missing, extra, or duplicate tasks. Ranks start at one, each task has at most
ten results, `frame_ms` is a non-negative integer measured from the start of
the cut clip, and selected results use unique videos within a task.

Validate the final output before submission:

```bash
"$PIPELINE_PYTHON" scripts/validate_submission.py \
  --tasks "$TASKS_PATH" \
  --submission "$ARTIFACT_ROOT/submission_top200/submission.json" \
  --shots-dir "$ARTIFACT_ROOT/omnishot_output/shots"
```

The command above uses the default output locations of `run_from_raw.sh`. If
you override `SHOT_OUTPUT` or `TASK_OUTPUT_ROOT`, use the corresponding paths
in the validation command.

## 10. Resuming and intermediate artifacts

This section describes generated state after the pipeline starts. It is not an
environment-installation section and it does not replace the checkpoint
download instructions in section 5.

Corpus embeddings are float16 NumPy shards with a manifest that fingerprints
the source JSONL, model, resizing, decoder, partitioning, and preprocessing
settings. Reranking similarly writes `run_config.json` and progress JSONL.
Restarting with a different resume-sensitive configuration is rejected.

Generated artifacts are intentionally excluded from Git. They are recreated by
the documented commands.

## 11. Reproducibility

The main inference code uses seed 2026, enables deterministic PyTorch
algorithms, disables cuDNN benchmarking and TF32, and uses deterministic frame
timestamps. The verified OmniShotCut environment reproduced the shot
boundaries of the three checked reference manifests exactly.

After the environments and model resources are prepared, `run_from_raw.sh`
creates predictions without interactive input. The pipeline uses no manual
labels, no per-task prediction edits, no paid API, and no external inference
service. Network access is used only by the documented setup and model-download
scripts before inference starts.

Small floating-point score differences may still occur across GPU
architectures, CUDA versions, or different model/library versions. They may
change ordering when two scores are nearly tied, so the pinned environments
are strongly recommended.

## 12. Known limitations and troubleshooting

- The official launcher requires two visible CUDA GPUs. It is not a one-GPU
  launcher.
- Each GPU must fit one Qwen3-VL-8B model in NF4 plus its runtime buffers.
  Twelve GiB per GPU was sufficient in the tested environment.
- TorchCodec 0.12 requires PyTorch 2.11 or newer. Install Torch and TorchCodec
  from the same CUDA wheel channel.
- The official path uses TorchCodec/NVDEC. Missing compatible FFmpeg libraries,
  NVDEC support, or a TorchCodec/PyTorch ABI mismatch will fail the decoder
  stage.
- CUDA 13 wheels require a driver that supports CUDA 13. Use the documented
  CUDA 12.6 wheel channel on a CUDA 12-only host.
- Model download requires access to Hugging Face and ModelScope. All large
  shards are checked after download; a checksum mismatch stops the script.
- OmniShotCut generation is time-consuming on the full corpus. It is resumable
  because existing per-video JSON files are skipped.
- Model inference is CUDA-only; CPU inference is not implemented for the two
  Qwen 8B stages.
