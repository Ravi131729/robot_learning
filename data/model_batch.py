"""Convert raw ABC batches into model-ready JAX policy batches."""

import jax.numpy as jnp

from data.hf_dino import encode_preprocessed_camera_images
from data.image_preprocessing import preprocess_camera_dict
from data.task_text import resolve_prompt
from data.types import PolicyBatch


def encode_abc_batch(
    batch,
    dino_encoder,
    task_embedder,
    camera_order=("top", "left", "right"),
    image_size=224,
):
    """Encode an `ABCBatch` into the policy's tensor interface.

    The returned shapes are:

    * `dino_tokens`: `(B, 197, 3, 768)`
    * `state`: `(B, 14)`
    * `task`: `(B, 512)`
    * `actions`: `(B, horizon, 14)`
    """
    processed_images = preprocess_camera_dict(
        batch.images,
        camera_order=camera_order,
        target_size=image_size,
    )
    dino_tokens = encode_preprocessed_camera_images(
        dino_encoder,
        processed_images,
    )
    prompts = [
        resolve_prompt(task_name, instruction)
        for task_name, instruction in zip(
            batch.task_names,
            batch.instructions,
        )
    ]
    task_embeddings = task_embedder.encode_prompts(prompts)

    model_batch = PolicyBatch(
        dino_tokens=jnp.asarray(dino_tokens),
        state=jnp.asarray(batch.state),
        task=jnp.asarray(task_embeddings),
        actions=jnp.asarray(batch.actions),
    )
    return model_batch.validate()
