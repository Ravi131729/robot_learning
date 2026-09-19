"""Timestep embeddings used by the flow-matching policy."""

import jax
import jax.numpy as jnp

from configs.robot_policy_config import DIT_DIM, TIME_EMBED_DIM
from robot_policy.common.layers import init_weight


def init_time_params(key):
    """Initialize the learned timestep projection parameters."""
    k1, k2 = jax.random.split(key)

    return {
        "w1": init_weight(k1, TIME_EMBED_DIM, DIT_DIM),
        "b1": jnp.zeros(DIT_DIM),
        "w2": init_weight(k2, DIT_DIM, DIT_DIM),
        "b2": jnp.zeros(DIT_DIM),
    }


def timestep_embedding(t, dim=TIME_EMBED_DIM):
    """Create sinusoidal embeddings for a batch of scalar timesteps."""
    half = dim // 2

    freqs = jnp.exp(
        -jnp.log(10000.0) * jnp.arange(half) / half
    )
    args = t[:, None] * freqs[None, :]

    return jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)


def time_embed(t, p):
    """Map scalar timesteps to DiT conditioning vectors."""
    x = timestep_embedding(t)
    x = jax.nn.silu(x @ p["w1"] + p["b1"])
    return x @ p["w2"] + p["b2"]
