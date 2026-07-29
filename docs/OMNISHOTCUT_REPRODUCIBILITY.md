# OmniShotCut reproducibility record

## Exact source and model

- Package: `omnishotcut 0.1.0`
- Git repository:
  `https://github.com/UVA-Computer-Vision-Lab/OmniShotCut`
- Git commit: `3331cd3163f7f17cd6d7c8fc12ffde22894ace01`
- Checkpoint repository: `uva-cv-lab/OmniShotCut`
- Checkpoint filename: `OmniShotCut_ckpt.pth`
- Hugging Face revision:
  `7f646c4ff4bb843e18c013481fb5d9ed2b068c6b`
- Checkpoint size: `164,149,963` bytes
- Checkpoint SHA-256:
  `5948ea78e00626c0e6c5e742e64873ef872cf4a5071d2a0841aed51c3e686cfa`

Auxiliary Torchvision ResNet18:

- Filename: `resnet18-f37072fd.pth`
- Size: `46,830,571` bytes
- SHA-256:
  `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`

## Inference settings

- Mode: `clean_shot`
- Overlap: 20 frames
- Overlap source: OmniShotCut's internal default argument
- The wrapper intentionally does not pass `overlap` in the official command.
- Shot frame indices are converted to milliseconds with
  `round(frame * 1000 / fps)`.

## Exact tested environment

- Base image:
  `cnstark/pytorch:2.3.1-py3.10.15-cuda12.1.0-devel-ubuntu22.04`
- Base image digest:
  `sha256:6386285f288cea3062933bde57d3db49550d86a78e1d7fb7c82efdcce3abe6a7`
- Python: `3.10.15`
- PyTorch: `2.3.1+cu121`
- Torchvision: `0.18.1+cu121`
- NumPy: `2.1.2`
- Pillow: `10.4.0`
- Decord: `0.6.0`
- OpenCV: `5.0.0.93`
- OpenCV headless: `5.0.0.93`
- FFmpeg: `7:4.4.2-0ubuntu0.22.04.1`

The recreated manifests matched the reference shot boundaries exactly for
`v3c1_00001`, `v3c1_00002`, and `v3c1_00004`. The absolute `video_path` field
was expected to differ between machines and is not part of shot-boundary
equivalence.

## Output schema

`tools/omnishot_pipeline.py` writes one UTF-8 JSON file per clip:

```json
{
  "video_id": "v3c1_00001",
  "video_path": "/machine-specific/path/00001.mp4",
  "fps": 25.0,
  "frame_count": 12345,
  "shots": [
    {
      "shot_id": 0,
      "start_frame": 0,
      "end_frame": 120,
      "start_ms": 0,
      "end_ms": 4800
    }
  ]
}
```

The same output directory also contains `run_config.json`, which records the
resolved checkpoint, checksum, source commit, environment versions, mode,
overlap, dataset roots, and shard settings.
