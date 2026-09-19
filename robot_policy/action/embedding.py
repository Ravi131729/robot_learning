"""Action and positional embeddings for the DiT policy."""

import jax
import jax.numpy as jnp

from configs.robot_policy_config import ACTION_DIM, ACTION_HORIZON, DIT_DIM
from robot_policy.common.layers import init_weight


def positional_embedding(length, dim):
    """Create sinusoidal positional embeddings for an action sequence."""
    half = dim // 2

    freqs = jnp.exp(
        -jnp.log(10000.0) * jnp.arange(half) / half
    )
    positions = jnp.arange(length)[:, None]
    args = positions * freqs[None, :]

    return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)


def init_action_params(key):
    """Initialize action projection and positional embedding parameters."""
    return {
        "w": init_weight(key, ACTION_DIM, DIT_DIM),
        "b": jnp.zeros(DIT_DIM),
        # Official pos_embed is a non-trainable (1, 30, 1536) parameter.
        "pos_embed": positional_embedding(ACTION_HORIZON, DIT_DIM)[None, :, :],
    }


def action_embed(actions, p):
    """Project action vectors into DiT tokens and add positions."""
    x = actions @ p["w"] + p["b"]
    return x + p["pos_embed"][:, : actions.shape[1], :]
