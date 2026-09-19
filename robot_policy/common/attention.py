"""Attention operators used by the DiT blocks."""

import jax
import jax.numpy as jnp

from configs.robot_policy_config import DIT_HEADS


def self_attention(x, p):
    """Apply multi-head self-attention over action tokens."""
    batch_size, token_count, dim = x.shape
    num_heads = DIT_HEADS
    head_dim = dim // num_heads

    query_weight, key_weight, value_weight = jnp.split(
        p["sa_qkv"], 3, axis=-1
    )
    query_bias, key_bias, value_bias = jnp.split(
        p["sa_qkv_bias"], 3, axis=-1
    )

    queries = (x @ query_weight + query_bias).reshape(
        batch_size, token_count, num_heads, head_dim
    ).transpose(0, 2, 1, 3)
    keys = (x @ key_weight + key_bias).reshape(
        batch_size, token_count, num_heads, head_dim
    ).transpose(0, 2, 1, 3)
    values = (x @ value_weight + value_bias).reshape(
        batch_size, token_count, num_heads, head_dim
    ).transpose(0, 2, 1, 3)

    scores = queries @ keys.transpose(0, 1, 3, 2)
    scores = scores / jnp.sqrt(head_dim)
    attention = jax.nn.softmax(scores, axis=-1)

    output = (attention @ values).transpose(
        0, 2, 1, 3
    ).reshape(batch_size, token_count, dim)

    return output @ p["sa_out"] + p["sa_out_bias"]


def cross_attention(x, vision, p):
    """Apply multi-head cross-attention from action tokens to vision tokens."""
    batch_size, token_count, dim = x.shape
    vision_token_count = vision.shape[1]
    num_heads = DIT_HEADS
    head_dim = dim // num_heads

    query_weight, key_weight, value_weight = jnp.split(
        p["ca_qkv"], 3, axis=-1
    )
    query_bias, key_bias, value_bias = jnp.split(
        p["ca_qkv_bias"], 3, axis=-1
    )

    queries = (x @ query_weight + query_bias).reshape(
        batch_size, token_count, num_heads, head_dim
    ).transpose(0, 2, 1, 3)
    keys = (vision @ key_weight + key_bias).reshape(
        batch_size, vision_token_count, num_heads, head_dim
    ).transpose(0, 2, 1, 3)
    values = (vision @ value_weight + value_bias).reshape(
        batch_size, vision_token_count, num_heads, head_dim
    ).transpose(0, 2, 1, 3)

    scores = queries @ keys.transpose(0, 1, 3, 2)
    scores = scores / jnp.sqrt(head_dim)
    attention = jax.nn.softmax(scores, axis=-1)

    output = (attention @ values).transpose(
        0, 2, 1, 3
    ).reshape(batch_size, token_count, dim)

    return output @ p["ca_out"] + p["ca_out_bias"]
