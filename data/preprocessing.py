"""Conversion from raw arrays to model-ready policy batches."""

import jax.numpy as jnp

from data.types import PolicyBatch


def prepare_policy_batch(
    dino_tokens,
    state,
    task,
    actions,
    state_stats=None,
    action_stats=None,
):
    """Convert raw arrays into a validated, optionally normalized PolicyBatch."""
    state = jnp.asarray(state)
    actions = jnp.asarray(actions)

    if state_stats is not None:
        state = state_stats.normalize(state)
    if action_stats is not None:
        actions = action_stats.normalize(actions)

    batch = PolicyBatch(
        dino_tokens=jnp.asarray(dino_tokens),
        state=state,
        task=jnp.asarray(task),
        actions=actions,
    )
    return batch.validate()
