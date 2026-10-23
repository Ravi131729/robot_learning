# Third-party code and assets

## ABC

The repository vendors the minimal ABC loader code required for standalone operation:

```text
third_party/abc_minimal/config.py
third_party/abc_minimal/dataloader.py
third_party/abc_minimal/dit.py
third_party/abc_minimal/operator.py
third_party/abc_minimal/preprocess.py
```

The source repository is [amazon-far/abc](https://github.com/amazon-far/abc). The copied files are documented in `third_party/abc_minimal/NOTICE.md`, and the upstream license is preserved in `third_party/ABC_LICENSE`.

`data/abc_official_loader.py` and `data/abc_official_batch.py` are local adapters between ABC's PyTorch batches and this repository's JAX policy.

## DINOv3

The image encoder is a locally stored Hugging Face DINOv3 ViT-B/16 checkpoint. The adapter removes DINO register tokens and exposes the 197-token representation expected by the policy.

## CLIP

The task encoder uses ABC's CLIP text interface and local CLIP assets supplied through `--clip-cache`. Task embeddings are 512-dimensional and frozen during policy training.

## Untracked assets

Datasets, model weights, checkpoints, W&B logs, and generated feature files are intentionally outside version control. Record their versions and checksums for published experiments.
