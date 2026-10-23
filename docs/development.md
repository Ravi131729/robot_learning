# Development guide

## Safe workflow

```bash
python -m py_compile train.py
python -m py_compile data/*.py robot_policy/**/*.py training/*.py
git diff --check
```

Run the debug preset before a long job whenever data loading or device placement changes.

## Directory ownership

```text
configs/       Shapes, constants, and architecture presets
data/          Dataset and encoder boundaries
robot_policy/  Policy computation and sampling
training/      Optimizer/update mechanics
train.py       Experiment orchestration and CLI
slurm/         Cluster launch descriptions
```

## Device rules

Async mode uses Torch on the encoder device and JAX on the policy device. Always print and verify both devices.

## Data rules

Do not commit datasets, model weights, checkpoints, W&B logs, or generated feature files. The repository is code- and documentation-focused.

## Review checklist

- cache paths exist;
- DINO and CLIP weights load;
- `nvidia-smi` shows the requested GPUs;
- JAX and Torch see CUDA;
- the debug smoke test completes;
- W&B run names identify the configuration;
- seed and batch size are recorded;
- logs and checkpoints are writable.
