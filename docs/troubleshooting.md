# Troubleshooting

## Dataset path points to `/home/ravi`

Pass explicit Palmetto paths:

```bash
--data-root "$HOME/robot_learning/cache"
--dino-model "$HOME/robot_learning/cache/dinov3-vitb16-pretrain-lvd1689m"
--clip-cache "$HOME/robot_learning/cache/clip"
```

## Missing `norm_stats.json`

```bash
ls "$HOME/robot_learning/cache/norm_stats.json"
```

## JAX falls back to CPU

```bash
python - <<'PY'
import jax, torch
print(jax.devices())
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
```

Run this inside the allocation, not only on the login node.

## TorchCodec or FFmpeg failure

TorchCodec must match the installed PyTorch version and must find the required FFmpeg shared libraries. Check versions and imports on a compute node.

## Wrong A100 VRAM variant

```bash
nvidia-smi
scontrol show job "$SLURM_JOB_ID" | grep -iE "constraint|gres|tres|node"
```

The reported GPU model and memory are authoritative.

## Async queue waits

The first wait includes producer startup and model loading. If steady-state queue wait remains high, increase `--prefetch-batches`, reduce batch size, optimize preprocessing, or use a faster encoder.

## GPU 1 utilization is low

Depth 1 and small batches finish too quickly. Test `medium` or `full` and measure samples/second. Increasing loader workers will not make the JAX model larger.

## Slurm output file is missing

```bash
mkdir -p logs
sbatch path/to/job.sbatch
```

## W&B failure

Use offline mode while debugging:

```bash
--wandb-mode offline
```
