"""A single AdaLN-gated Diffusion Transformer block."""

import jax
import jax.numpy as jnp

from configs.robot_policy_config import DIT_DIM, DIT_MLP_DIM
from robot_policy.common.attention import cross_attention, self_attention
from robot_policy.common.layers import init_weight, layer_norm, modulate


def init_dit_block(key):
    """Initialize parameters for one DiT block."""
    k = jax.random.split(key, 11)

    return {
        # 9 AdaLN vectors
        "ada_w": init_weight(k[0], DIT_DIM, 9 * DIT_DIM),
        "ada_b": jnp.zeros(9 * DIT_DIM),
        # Packed attention projections and explicit biases
        "sa_qkv": init_weight(k[1], DIT_DIM, 3 * DIT_DIM),
        "sa_qkv_bias": jnp.zeros(3 * DIT_DIM),
        "sa_out": init_weight(k[4], DIT_DIM, DIT_DIM),
        "sa_out_bias": jnp.zeros(DIT_DIM),
        "ca_qkv": init_weight(k[5], DIT_DIM, 3 * DIT_DIM),
        "ca_qkv_bias": jnp.zeros(3 * DIT_DIM),
        "ca_out": init_weight(k[8], DIT_DIM, DIT_DIM),
        "ca_out_bias": jnp.zeros(DIT_DIM),
        # MLP
        "mlp_up": init_weight(k[9], DIT_DIM, DIT_MLP_DIM),
        "mlp_up_bias": jnp.zeros(DIT_MLP_DIM),
        "mlp_down": init_weight(k[10], DIT_MLP_DIM, DIT_DIM),
        "mlp_down_bias": jnp.zeros(DIT_DIM),
    }


def dit_block(x, vision, cond, p):
    """Apply self-attention, vision cross-attention, and an MLP."""
    ada_condition = jax.nn.silu(cond)
    ada_condition = ada_condition @ p["ada_w"] + p["ada_b"]

    shift_1, scale_1, gate_1, shift_2, scale_2, gate_2, shift_3, scale_3, gate_3 = (
        jnp.split(ada_condition, 9, axis=-1)
    )

    # Self-attention
    hidden = layer_norm(x, eps=1e-6)
    hidden = modulate(hidden, shift_1, scale_1)
    x = x + gate_1[:, None, :] * self_attention(hidden, p)

    # Vision cross-attention
    hidden = layer_norm(x, eps=1e-6)
    hidden = modulate(hidden, shift_2, scale_2)
    normalized_vision = layer_norm(vision, eps=1e-6)
    x = x + gate_2[:, None, :] * cross_attention(hidden, normalized_vision, p)

    # MLP
    hidden = layer_norm(x, eps=1e-6)
    hidden = modulate(hidden, shift_3, scale_3)
    hidden = jax.nn.gelu(
        hidden @ p["mlp_up"] + p["mlp_up_bias"],
        approximate=True,
    )
    hidden = hidden @ p["mlp_down"] + p["mlp_down_bias"]

    return x + gate_3[:, None, :] * hidden
