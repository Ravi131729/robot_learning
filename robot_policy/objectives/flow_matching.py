"""Flow-matching objective for action-velocity prediction."""

import jax
import jax.numpy as jnp

from robot_policy.policy import policy_forward


def flow_matching_loss(
    params,
    dino_tokens,
    state,
    task,
    actions,
    key,
):
    """Compute flow-matching MSE for a batch of demonstrated actions."""
    batch_size = actions.shape[0]
    time_key, noise_key = jax.random.split(key)

    time = jax.random.uniform(time_key, (batch_size,))
    noise = jax.random.normal(noise_key, actions.shape)

    time_3d = time[:, None, None]
    noisy_actions = (1.0 - time_3d) * actions + time_3d * noise
    target_velocity = noise - actions

    predicted_velocity = policy_forward(
        dino_tokens,
        state,
        task,
        noisy_actions,
        time,
        params,
    )

    return jnp.mean((predicted_velocity - target_velocity) ** 2)
