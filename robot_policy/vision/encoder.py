"""Camera-specific vision encoding for the DiT policy."""

import jax
import jax.numpy as jnp

from configs.robot_policy_config import DINO_DIM, DIT_DIM, NUM_CAMERAS
from robot_policy.common.layers import init_weight
from robot_policy.vision.pooler import init_pooler, vision_pool_block


def init_vision_params(key):
    """Initialize parameters for all camera vision encoders."""
    k = jax.random.split(key, 5)

    return {
        "poolers": {
            "top": init_pooler(k[0]),
            "left": init_pooler(k[1]),
            "right": init_pooler(k[2]),
        },
        "vision_to_dit": init_weight(k[3], DINO_DIM, DIT_DIM),
        "vision_to_dit_bias": jnp.zeros(DIT_DIM),
        # Retained for exact official state-tree coverage; official forward
        # keeps this compatibility projection unused.
        "img_proj": init_weight(k[3], DINO_DIM, DIT_DIM),
        "img_proj_bias": jnp.zeros(DIT_DIM),
        "camera_embed": 0.02 * jax.random.normal(
            k[4], (NUM_CAMERAS, DIT_DIM)
        ),
    }


def encode_camera(dino_tokens, camera_name, camera_id, p):
    """Encode one camera's DINO tokens into DiT feature tokens."""
    x = vision_pool_block(dino_tokens, p["poolers"][camera_name])
    x = x @ p["vision_to_dit"] + p["vision_to_dit_bias"]
    return x + p["camera_embed"][camera_id]


def encode_vision_for_dit(dino_tokens, p):
    """Encode and concatenate the top, left, and right camera views."""
    names = ("top", "left", "right")

    views = [
        encode_camera(dino_tokens[:, :, camera_id, :], name, camera_id, p)
        for camera_id, name in enumerate(names)
    ]

    return jnp.concatenate(views, axis=1)
