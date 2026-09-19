"""Final AdaLN and action-velocity output layer."""

import jax
import jax.numpy as jnp

from robot_policy.common.layers import layer_norm, modulate


def final_layer(x, cond, p):
    """Convert final DiT token features into action velocities."""
    ada_condition = jax.nn.silu(cond)
    ada_condition = ada_condition @ p["final_ada_w"] + p["final_ada_b"]

    shift, scale = jnp.split(ada_condition, 2, axis=-1)
    x = layer_norm(x, eps=1e-6)
    x = modulate(x, shift, scale)

    return x @ p["final_w"] + p["final_b"]
