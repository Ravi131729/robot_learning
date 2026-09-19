# Robot Learning Policy

Modular JAX implementation of a vision-language-action flow-matching policy
for the ABC simulation benchmark. The policy consumes three camera views,
robot state, and a CLIP task embedding, then predicts 30-step action chunks
with a DiT-style transformer.

## Data and ABC integration

The local dataset is expected at:

```text
/home/ravi/robot_learning/cache
```

The cache is ignored by Git. Training uses ABC's official
`EpisodeDataset`, `collate`, worker-based `DataLoader`, normalization, and
TorchCodec video decoding from the upstream repository:

```text
https://github.com/amazon-far/abc
```

The files in `data/abc_official_loader.py` and
`data/abc_official_batch.py` are only integration adapters between ABC's
PyTorch batches and this repository's JAX policy.

## Environment

Use the `base` environment with JAX CUDA, PyTorch CUDA, Transformers, Optax,
TorchCodec, and a local checkout of the upstream ABC repository.

```bash
git clone https://github.com/amazon-far/abc.git
export ABC_REPO_ROOT=/path/to/abc
```

TorchCodec must match PyTorch. For the current PyTorch 2.10 installation:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  torchcodec==0.10.0
```

Verify GPU access:

```bash
python - <<'PY'
import jax
import torch
print(jax.devices())
print(torch.cuda.is_available())
PY
```

## Training

Run a small real training experiment:

```bash
python train.py \
  --preset small \
  --batch-size 4 \
  --num-workers 4 \
  --steps 1000 \
  --val-steps 20 \
  --sample-steps 10 \
  --timing \
  --memory
```

Presets are:

```text
debug: depth 1
small: depth 8
full:  depth 32
```

Checkpoints are written to `checkpoints/`, which is ignored by Git. The
validation loss uses `val_sim`, and sampled actions are reported in ABC's
normalized action space; unnormalize them before sending them to the
simulator.

## Repository layout

```text
configs/       Architecture constants and presets
data/          ABC adapters, DINOv3/CLIP encoding, batch types
robot_policy/  Vision conditioning, DiT, objective, and sampler
training/      JAX optimizer state and update loop
train.py       Command-line training, validation, timing, and checkpoints
```

ABC remains the source of truth for dataset semantics and loading. This
repository owns the JAX policy architecture and its training adapters.

## ABC attribution

This project uses and adapts the official ABC data-loading interfaces from
[amazon-far/abc](https://github.com/amazon-far/abc), including its
`EpisodeDataset`, collation, normalization, and TorchCodec video-decoding
path. Please cite the ABC project:

```bibtex
@misc{abc2026,
  title = {Scalable Behavior Cloning with Open Data, Training, and Evaluation},
  author = {Allshire et al.},
  year = {2026},
  url = {https://github.com/amazon-far/abc}
}
```
