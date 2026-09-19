"""Euler sampler for the policy's flow-matching velocity field."""

import jax
import jax.numpy as jnp

from configs.robot_policy_config import ACTION_DIM, ACTION_HORIZON
from robot_policy.policy import policy_forward


def sample_actions(
    dino_tokens,
    state,
    task,
    params,
    key,
    num_steps=10,
    initial_noise=None,
):
    """Generate an action chunk by integrating from noise to actions.

    Training defines the interpolation as x_t = (1-t)x_0 + t*noise,
    so inference starts at t=1 and integrates backward toward t=0.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")

    if initial_noise is None:
        batch_size = dino_tokens.shape[0]
        initial_noise = jax.random.normal(
            key,
            (batch_size, ACTION_HORIZON, ACTION_DIM),
        )

    action_state = initial_noise
    time_grid = jnp.linspace(1.0, 0.0, num_steps + 1)

    for step in range(num_steps):
        time_start = time_grid[step]
        time_end = time_grid[step + 1]
        dt = time_end - time_start
        time = jnp.full((action_state.shape[0],), time_start)

        velocity = policy_forward(
            dino_tokens,
            state,
            task,
            action_state,
            time,
            params,
        )
        action_state = action_state + dt * velocity

    return action_state
