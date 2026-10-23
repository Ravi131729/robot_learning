# Profiling and utilization

## Timed stages

With `--timing`, the training loop reports dataset/video loading, DINOv3 encoding, CLIP encoding, CPU-to-GPU preparation, JAX compilation/update, and queue wait in async mode.

The first JAX update includes compilation and should not be used as steady-state timing.

## Throughput

```text
samples/sec = batch_size / steady_state_step_time
```

For async training, include both producer and consumer behavior. A fast JAX update is not useful if the queue is frequently empty.

## Why utilization fluctuates

The pipeline switches between CPU loading, Torch DINO kernels, text encoding, host/device transfers, and JAX kernels. Dashboard sampling makes short bursts appear as fluctuations. A small depth-1 policy also finishes quickly and leaves gaps between updates.

Increasing `num_workers` only helps while data loading is the bottleneck. Larger batch sizes and deeper DiT presets increase useful policy computation.

## Memory accounting

The memory output distinguishes process RSS, Torch allocated/reserved/peak memory, JAX allocated memory, and parameter-size estimates. Frozen DINO and CLIP parameters still occupy memory even though they are not optimized.

## Recommended profile

Run at least 200–500 steps and ignore startup:

```bash
python train.py --preset medium --batch-size 16 --steps 500 \
  --pipeline-mode async --encoder-device cuda:0 --policy-device cuda:1 \
  --timing --memory
```

Use an Nsight Systems trace only after application-level timings identify a real bottleneck.
