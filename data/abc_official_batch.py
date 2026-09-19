"""Convert batches from ABC's official PyTorch loader to ``PolicyBatch``.

The ABC loader remains responsible for episode sampling, normalization, video
decoding, workers, and collation. This file is the local JAX-facing adapter.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import torch

from data.types import PolicyBatch


def encode_official_batch(batch, dino, clip, device, with_timings=False):
    timings = {}
    images = batch["images"]
    camera_names = tuple(images)
    stacked = torch.stack([images[name] for name in camera_names], dim=1)
    batch_size, camera_count = stacked.shape[:2]

    start = time.perf_counter()
    flattened = stacked.reshape(batch_size * camera_count, 3, 224, 224)
    dino_tokens = dino.encode_image_tokens(flattened)
    dino_tokens = (
        dino_tokens.reshape(batch_size, camera_count, 197, 768)
        .permute(0, 2, 1, 3)
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    if with_timings:
        timings["DINOv3 encoding"] = time.perf_counter() - start

    start = time.perf_counter()
    task_embeddings = clip.encode_prompts(batch["prompt"])
    if with_timings:
        timings["CLIP encoding"] = time.perf_counter() - start

    start = time.perf_counter()
    model_batch = PolicyBatch(
        dino_tokens=jnp.asarray(dino_tokens),
        state=jnp.asarray(batch["state"].numpy(), dtype=jnp.float32),
        task=jnp.asarray(task_embeddings),
        actions=jnp.asarray(batch["actions"].numpy(), dtype=jnp.float32),
    ).validate()
    model_batch = PolicyBatch(
        dino_tokens=jax.device_put(model_batch.dino_tokens, device),
        state=jax.device_put(model_batch.state, device),
        task=jax.device_put(model_batch.task, device),
        actions=jax.device_put(model_batch.actions, device),
    )
    for value in (model_batch.dino_tokens, model_batch.state,
                  model_batch.task, model_batch.actions):
        value.block_until_ready()
    if with_timings:
        timings["CPU-to-GPU batch preparation"] = time.perf_counter() - start
    return model_batch, timings
