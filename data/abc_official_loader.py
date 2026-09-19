"""Thin bridge to the vendored ABC worker-based dataloader.

The imported files are copied from the Amazon FAR ABC repository:
https://github.com/amazon-far/abc
"""

from pathlib import Path
import json
import sys
from functools import partial

import torch
from torch.utils.data import DataLoader

from data.abc_paths import resolve_abc_root


def _add_abc_to_path(abc_root):
    abc_root = resolve_abc_root(abc_root)
    root = str(abc_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    return abc_root


def _infer_task_name(split_dir):
    for episode_dir in sorted(Path(split_dir).iterdir()):
        metadata_path = episode_dir / "episode_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if metadata.get("task_name"):
                return metadata["task_name"]
    raise ValueError(f"could not infer task name from {split_dir}")


def create_abc_loader(
    cache_root,
    split="train_sim",
    batch_size=1,
    num_workers=4,
    abc_root=None,
    seed=0,
    train=True,
):
    """Create ABC's official `EpisodeDataset` and worker DataLoader."""
    abc_root = _add_abc_to_path(abc_root)
    from abc_minimal.config import DiTConfig
    from abc_minimal.dataloader import EpisodeDataset, collate
    from abc_minimal.preprocess import load_norm_stats

    split_dir = Path(cache_root) / split
    norm_stats = load_norm_stats(Path(cache_root) / "norm_stats.json")
    model_config = DiTConfig()
    dataset = EpisodeDataset(
        data_dir=split_dir,
        norm_stats=norm_stats,
        train=train,
        default_task_name=_infer_task_name(split_dir),
        mask_state_ratio=0.0,
        model_config=model_config,
        norm_preset="imagenet",
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        collate_fn=partial(collate, camera_keys=dataset.camera_keys),
        pin_memory=True,
        drop_last=train,
        persistent_workers=num_workers > 0,
        generator=generator,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
    return loader
