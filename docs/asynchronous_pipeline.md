# Asynchronous two-GPU pipeline

## Purpose

DINOv3 and CLIP are frozen, but inference still consumes GPU time. Async mode isolates frozen encoder work from the trainable policy update.

```text
Producer process, GPU 0
  ABC loader → augmentation → DINOv3 → CLIP → bounded queue
                                                       │
Consumer process, GPU 1                            queue.get()
  JAX device transfer → policy update → loss
```

This is not persistent feature caching. Every batch is encoded again, so fresh training augmentation is preserved.

## Configuration

```bash
--pipeline-mode async
--encoder-device cuda:0
--policy-device cuda:1
--prefetch-batches 4
```

The queue is bounded. A full queue makes the producer wait; an empty queue makes the policy process wait.

## Timing interpretation

- `producer total`: loader batch encoding and host preparation;
- `DINOv3 encoding`: frozen image encoder time;
- `CLIP encoding`: text encoder time;
- `CPU-to-GPU batch preparation`: policy-device transfer;
- `feature queue wait`: time waiting for the next producer batch;
- `JAX update`: policy compilation or update time.

The first queue wait is often large because the producer is loading weights and filling the queue. Ignore startup when comparing steady-state throughput.

If policy updates are much faster than producer batches, the queue will eventually drain. Increase policy depth or batch size, optimize DINO, or use a faster encoder. If queue wait is near zero, the producer is hiding its latency successfully.

## Validation behavior

The producer is paused before validation and restarted afterward. This avoids duplicate DINO/CLIP models competing for GPU 0 memory, but restarting the producer makes validation more expensive.

## Data transport

The initial transport path is:

```text
GPU 0 features → host memory → GPU 1 JAX arrays
```

This is intentionally simple and debuggable. Direct GPU-to-GPU transfer should only be considered after measuring host-transfer time.
