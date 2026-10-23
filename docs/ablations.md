# Ablation plan

## Suggested sweep

| Preset | Depth | Suggested batches |
|---|---:|---:|
| `small` | 8 | 64, 96, 128 |
| `medium` | 16 | 16, 32, 64 |
| `full` | 32 | 16, 32, 64 on A100 80 GB |

The maximum batch depends on activation memory. Stop before out-of-memory and record peak memory.

## Record for every run

```text
preset and depth
batch size
number of workers
pipeline mode
encoder and policy GPU types
seed
training steps
validation loss
samples/sec
steady-state JAX update time
DINO time
queue wait time
peak memory
```

## Experimental order

1. Validate the async smoke test.
2. Compare inline and async mode with the same small configuration.
3. Sweep batch size for depth 8.
4. Sweep depth 16 on A100 40 GB.
5. Sweep depth 32 on A100 80 GB.
6. Repeat the best configuration with three seeds.
7. Compare backbones only after the training pipeline is stable.

Prefer validation loss and samples/second over instantaneous GPU utilization alone.

## W&B naming

```text
a100-40gb-depth16-bs16
a100-80gb-depth32-bs64-seed1
```
