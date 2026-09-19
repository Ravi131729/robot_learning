"""Feature and action normalization utilities."""

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class NormalizationStats:
    """Per-feature statistics used for stable policy training and inference."""

    mean: jnp.ndarray
    std: jnp.ndarray
    eps: float = 1e-6

    def normalize(self, values):
        return (values - self.mean) / jnp.maximum(self.std, self.eps)

    def denormalize(self, values):
        return values * jnp.maximum(self.std, self.eps) + self.mean


def fit_normalization_stats(values, eps=1e-6):
    """Fit per-final-dimension statistics from a batch or trajectory array."""
    values = jnp.asarray(values)
    if values.ndim < 2:
        raise ValueError("values must have at least batch and feature dimensions")

    reduction_axes = tuple(range(values.ndim - 1))
    mean = jnp.mean(values, axis=reduction_axes)
    std = jnp.std(values, axis=reduction_axes)
    return NormalizationStats(mean=mean, std=std, eps=eps)
