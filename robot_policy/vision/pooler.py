"""Camera-specific cross-attention pooling for vision tokens."""

import jax
import jax.numpy as jnp

from configs.robot_policy_config import (
    DINO_DIM,
    NUM_POOL_HEADS,
    NUM_POOL_QUERIES,
    POOL_MLP_DIM,
)
from robot_policy.common.layers import init_weight, layer_norm


def init_pooler(key):
    """Initialize parameters for one camera's vision pooler."""
    k = jax.random.split(key, 7)

    return {
        "pool_queries": 0.02 * jax.random.normal(
            k[0], (NUM_POOL_QUERIES, DINO_DIM)
        ),
        "qkv_weight": init_weight(k[1], DINO_DIM, 3 * DINO_DIM),
        "qkv_bias": jnp.zeros(3 * DINO_DIM),
        "out_weight": init_weight(k[4], DINO_DIM, DINO_DIM),
        "out_bias": jnp.zeros(DINO_DIM),
        "ffn_up": init_weight(k[5], DINO_DIM, POOL_MLP_DIM),
        "ffn_up_bias": jnp.zeros(POOL_MLP_DIM),
        "ffn_down": init_weight(k[6], POOL_MLP_DIM, DINO_DIM),
        "ffn_down_bias": jnp.zeros(DINO_DIM),
        "pool_ln_scale": jnp.ones(DINO_DIM),
        "pool_ln_bias": jnp.zeros(DINO_DIM),
        "ffn_ln_scale": jnp.ones(DINO_DIM),
        "ffn_ln_bias": jnp.zeros(DINO_DIM),
    }


def cross_attention_pool(vision_tokens, p):
    """Pool a sequence of vision tokens into learned query tokens."""
    batch_size, token_count, dim = vision_tokens.shape
    num_heads = NUM_POOL_HEADS
    head_dim = dim // num_heads

    normalized_tokens = layer_norm(
        vision_tokens,
        p["pool_ln_scale"],
        p["pool_ln_bias"],
        eps=1e-5,
    )
    normalized_queries = layer_norm(
        p["pool_queries"],
        p["pool_ln_scale"],
        p["pool_ln_bias"],
        eps=1e-5,
    )

    query_weight, key_weight, value_weight = jnp.split(
        p["qkv_weight"], 3, axis=-1
    )
    query_bias, key_bias, value_bias = jnp.split(
        p["qkv_bias"], 3, axis=-1
    )

    queries = (normalized_queries @ query_weight + query_bias).reshape(
        1, NUM_POOL_QUERIES, num_heads, head_dim
    ).transpose(0, 2, 1, 3)
    keys = (normalized_tokens @ key_weight + key_bias).reshape(
        batch_size, token_count, num_heads, head_dim
    ).transpose(0, 2, 1, 3)
    values = (normalized_tokens @ value_weight + value_bias).reshape(
        batch_size, token_count, num_heads, head_dim
    ).transpose(0, 2, 1, 3)

    scores = queries @ keys.transpose(0, 1, 3, 2)
    scores = scores / jnp.sqrt(head_dim)
    attention = jax.nn.softmax(scores, axis=-1)

    pooled = (attention @ values).transpose(
        0, 2, 1, 3
    ).reshape(batch_size, NUM_POOL_QUERIES, dim)

    return pooled @ p["out_weight"] + p["out_bias"]


def vision_pool_block(vision_tokens, p):
    """Apply vision cross-attention pooling followed by an FFN residual."""
    x = cross_attention_pool(vision_tokens, p)
    residual = x

    x = layer_norm(
        x,
        p["ffn_ln_scale"],
        p["ffn_ln_bias"],
        eps=1e-5,
    )
    x = jax.nn.gelu(x @ p["ffn_up"] + p["ffn_up_bias"])
    x = x @ p["ffn_down"] + p["ffn_down_bias"]

    return residual + x
