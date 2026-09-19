"""Shared JAX layer primitives used by the robot policy."""

import jax
import jax.numpy as jnp


def init_weight(key, in_dim, out_dim, scale=0.02):
    """Initialize a dense weight matrix with a normal distribution."""
    return scale * jax.random.normal(key, (in_dim, out_dim))


def layer_norm(x, scale=None, bias=None, eps=1e-6):
    """Apply layer normalization over the final dimension."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    y = (x - mean) / jnp.sqrt(var + eps)

    if scale is not None:
        y = y * scale
    if bias is not None:
        y = y + bias

    return y


def modulate(x, shift, scale):
    """Apply per-example shift and scale to token features."""
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]
