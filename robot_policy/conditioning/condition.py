"""State, task, and timestep conditioning for the DiT policy."""

import jax
import jax.numpy as jnp

from configs.robot_policy_config import DIT_DIM, STATE_DIM, TASK_DIM
from robot_policy.common.layers import init_weight, layer_norm
from robot_policy.conditioning.timestep import time_embed


def init_condition_params(key):
    """Initialize state, task, and conditioning MLP parameters."""
    k = jax.random.split(key, 4)

    return {
        "state_w": init_weight(k[0], STATE_DIM, DIT_DIM),
        "state_b": jnp.zeros(DIT_DIM),
        "task_w": init_weight(k[1], TASK_DIM, DIT_DIM),
        "task_b": jnp.zeros(DIT_DIM),
        "cond_w1": init_weight(k[2], 3 * DIT_DIM, DIT_DIM),
        "cond_b1": jnp.zeros(DIT_DIM),
        "cond_w2": init_weight(k[3], DIT_DIM, DIT_DIM),
        "cond_b2": jnp.zeros(DIT_DIM),
        "cond_ln_scale": jnp.ones(DIT_DIM),
        "cond_ln_bias": jnp.zeros(DIT_DIM),
    }


def condition_embed(state, task, t, p, time_p):
    """Combine state, task, and timestep features into one condition vector."""
    state_emb = state @ p["state_w"] + p["state_b"]
    task_emb = task @ p["task_w"] + p["task_b"]
    time_emb = time_embed(t, time_p)

    x = jnp.concatenate([state_emb, task_emb, time_emb], axis=-1)
    x = jax.nn.silu(x @ p["cond_w1"] + p["cond_b1"])
    x = x @ p["cond_w2"] + p["cond_b2"]

    return layer_norm(
        x,
        p["cond_ln_scale"],
        p["cond_ln_bias"],
        eps=1e-5,
    )
