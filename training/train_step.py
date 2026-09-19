"""JAX/Optax training primitives for the flow-matching policy."""

from typing import Any, NamedTuple

import jax
import optax

from robot_policy.objectives.flow_matching import flow_matching_loss


class TrainState(NamedTuple):
    """Parameters and optimizer state carried between training steps."""

    step: Any
    params: Any
    opt_state: Any


def create_optimizer(learning_rate, weight_decay=0.0):
    """Create the default AdamW optimizer."""
    return optax.adamw(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )


def create_train_state(params, optimizer):
    """Create an initialized training state for model parameters."""
    return TrainState(
        step=0,
        params=params,
        opt_state=optimizer.init(params),
    )


def make_train_step(optimizer):
    """Build a compiled one-step flow-matching update function."""

    @jax.jit
    def train_step(train_state, dino_tokens, state, task, actions, key):
        def loss_fn(params):
            return flow_matching_loss(
                params,
                dino_tokens,
                state,
                task,
                actions,
                key,
            )

        loss, gradients = jax.value_and_grad(loss_fn)(train_state.params)
        updates, new_opt_state = optimizer.update(
            gradients,
            train_state.opt_state,
            train_state.params,
        )
        new_params = optax.apply_updates(train_state.params, updates)

        new_train_state = TrainState(
            step=train_state.step + 1,
            params=new_params,
            opt_state=new_opt_state,
        )
        return new_train_state, loss

    return train_step
