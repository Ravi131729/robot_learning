"""Feature and action normalization utilities."""

from dataclasses import dataclass
import json
from pathlib import Path

import jax.numpy as jnp


@dataclass
class NormalizationStats:
    """Per-feature statistics used for stable policy training and inference."""

    mean: jnp.ndarray
    std: jnp.ndarray
    eps: float = 1e-6

    def normalize(self, values):
        return (values - self.mean) / (self.std + self.eps)

    def denormalize(self, values):
        return values * (self.std + self.eps) + self.mean


def fit_normalization_stats(values, eps=1e-6):
    """Fit per-final-dimension statistics from a batch or trajectory array."""
    values = jnp.asarray(values)
    if values.ndim < 2:
        raise ValueError("values must have at least batch and feature dimensions")

    reduction_axes = tuple(range(values.ndim - 1))
    mean = jnp.mean(values, axis=reduction_axes)
    std = jnp.std(values, axis=reduction_axes)
    return NormalizationStats(mean=mean, std=std, eps=eps)


def parse_abc_norm_stats(raw, eps=1e-6):
    """Parse ABC's ``norm_stats.json`` into reusable feature statistics.

    ABC stores statistics under ``norm_stats``. Older files may have an
    additional dataset key, so this follows the official fallback behavior.
    Quantiles are retained in the source file for clipping policies but are
    not needed for standard z-score normalization.
    """
    stats = raw.get("norm_stats", raw)
    if "state" not in stats or "actions" not in stats:
        dataset_key = "xdof" if "xdof" in stats else next(iter(stats))
        stats = stats[dataset_key]

    return {
        name: NormalizationStats(
            mean=jnp.asarray(values["mean"], dtype=jnp.float32),
            std=jnp.asarray(values["std"], dtype=jnp.float32),
            eps=eps,
        )
        for name, values in (
            ("state", stats["state"]),
            ("actions", stats["actions"]),
        )
    }


def load_abc_norm_stats(path, eps=1e-6):
    """Load ABC normalization statistics from a JSON file."""
    raw = json.loads(Path(path).read_text())
    return parse_abc_norm_stats(raw, eps=eps)
