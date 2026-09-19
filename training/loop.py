"""Small reusable training loops built on the compiled train step."""

import jax
import numpy as np


def run_train_steps(train_state, batch, train_step, key, num_steps):
    """Run repeated updates on one or more model-ready batches.

    This helper intentionally stays agnostic to dataset loading. A later
    trainer can provide a different batch at each iteration while reusing the
    same train-step contract.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")

    losses = []
    for step in range(num_steps):
        step_key = jax.random.fold_in(key, step)
        train_state, loss = train_step(train_state, batch, step_key)
        losses.append(float(loss))

    return train_state, np.asarray(losses, dtype=np.float32)
