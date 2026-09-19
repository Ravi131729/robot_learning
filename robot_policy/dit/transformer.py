"""Full Diffusion Transformer stack."""

import jax
import jax.numpy as jnp

from configs.robot_policy_config import ACTION_DIM, DIT_DEPTH, DIT_DIM
from robot_policy.common.layers import init_weight
from robot_policy.dit.block import dit_block, init_dit_block
from robot_policy.dit.output import final_layer


def init_dit_params(key, depth=DIT_DEPTH):
    """Initialize all DiT blocks and the final output layer."""
    if depth < 1:
        raise ValueError("DiT depth must be at least 1")

    keys = jax.random.split(key, depth + 2)

    return {
        "blocks": [
            init_dit_block(keys[i])
            for i in range(depth)
        ],
        "final_ada_w": init_weight(keys[-2], DIT_DIM, 2 * DIT_DIM),
        "final_ada_b": jnp.zeros(2 * DIT_DIM),
        "final_w": init_weight(keys[-1], DIT_DIM, ACTION_DIM),
        "final_b": jnp.zeros(ACTION_DIM),
    }


def dit_forward(action_tokens, vision_tokens, cond, p):
    """Run action tokens through the full DiT stack."""
    x = action_tokens

    for block_params in p["blocks"]:
        x = dit_block(x, vision_tokens, cond, block_params)

    return final_layer(x, cond, p)
