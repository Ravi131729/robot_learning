"""Typed data containers exchanged by the policy pipeline."""

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class PolicyBatch:
    """A normalized batch consumed by the policy and flow objective."""

    dino_tokens: jnp.ndarray
    state: jnp.ndarray
    task: jnp.ndarray
    actions: jnp.ndarray

    @property
    def batch_size(self):
        return self.actions.shape[0]

    @property
    def action_horizon(self):
        return self.actions.shape[1]

    def validate(self):
        """Validate batch dimensions before sending data to the model."""
        if self.dino_tokens.ndim != 4:
            raise ValueError("dino_tokens must have shape (B, tokens, cameras, dim)")
        if self.state.ndim != 2:
            raise ValueError("state must have shape (B, state_dim)")
        if self.task.ndim != 2:
            raise ValueError("task must have shape (B, task_dim)")
        if self.actions.ndim != 3:
            raise ValueError("actions must have shape (B, horizon, action_dim)")

        batch_sizes = {
            self.dino_tokens.shape[0],
            self.state.shape[0],
            self.task.shape[0],
            self.actions.shape[0],
        }
        if len(batch_sizes) != 1:
            raise ValueError("all policy inputs must have the same batch size")

        return self
