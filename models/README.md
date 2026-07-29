# Model files

Run:

```bash
./scripts/download_models.sh
./scripts/download_omnishot_models.sh
```

Expected layout:

```text
models/
├── Qwen3-VL-Embedding-8B/
│   ├── model-00001-of-00004.safetensors
│   ├── model-00002-of-00004.safetensors
│   ├── model-00003-of-00004.safetensors
│   ├── model-00004-of-00004.safetensors
│   └── tokenizer/config files
├── Qwen3-VL-Reranker-8B/
│   ├── model-00001-of-00004.safetensors
│   ├── model-00002-of-00004.safetensors
│   ├── model-00003-of-00004.safetensors
│   ├── model-00004-of-00004.safetensors
│   └── tokenizer/config files
├── OmniShotCut/
│   └── OmniShotCut_ckpt.pth
└── torch/
    └── hub/checkpoints/
        └── resnet18-f37072fd.pth
```

## Qwen weight checksums

| Model/file | Bytes | SHA-256 |
|---|---:|---|
| Embedding `model-00001-of-00004.safetensors` | 4,998,056,552 | `79ef275ec5f751d5fb59357c00d473268f9fd74abf5e38aa30137d268e7733c4` |
| Embedding `model-00002-of-00004.safetensors` | 4,915,962,464 | `a4da61f512e84fc0f0b80bcb7bcc5137eb3bf25b658a7d55f84f4056078545f0` |
| Embedding `model-00003-of-00004.safetensors` | 4,915,962,496 | `7fb17cf8f06d6fe5aaacf114e85c4e6d8318799f24b517f50d1ec154a8d47007` |
| Embedding `model-00004-of-00004.safetensors` | 1,459,698,112 | `000213b6d1d03ed9023fac23716da51ec4c5be221a04526c2f732d31d8fed1f5` |
| Reranker `model-00001-of-00004.safetensors` | 4,998,056,552 | `00db2779f4c81c18a551b05ee617a4012af2601ec47181c15629ae756ef367d6` |
| Reranker `model-00002-of-00004.safetensors` | 4,915,962,464 | `6ef4ccabf1f72c42eed016adca6d46528e66875171268e5ff603c1df2f97fa3d` |
| Reranker `model-00003-of-00004.safetensors` | 4,915,962,496 | `f036e887d2b27b56a3b22fe200bbdd7e0c8f075100f87bd2b6653fa8fca02973` |
| Reranker `model-00004-of-00004.safetensors` | 2,704,357,976 | `723d600bc3947051da769a35fb7bb62419d60d558bee7e9c9bb15919ccc79190` |
