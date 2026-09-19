"""Loader for ABC's official OpenAI CLIP text embedder."""

from pathlib import Path
import sys

from data.task_text import TaskTextEmbedder
from data.abc_paths import resolve_abc_root


def load_abc_clip_text_embedder(
    abc_root=None,
    cache_dir="/home/ravi/robot_learning/cache/clip",
    device="cpu",
):
    """Load ABC's CLIP ViT-B/32 text encoder from local assets."""
    abc_root = resolve_abc_root(abc_root)

    abc_root_string = str(abc_root)
    if abc_root_string not in sys.path:
        sys.path.insert(0, abc_root_string)

    try:
        from abc_minimal.config import ClipConfig
        from abc_minimal.dit import CLIPTextEmbedder
    except ImportError as exc:
        raise ImportError(
            "Could not import ABC's CLIP text encoder. "
            "Install the dependencies from the upstream ABC repository."
        ) from exc

    config = ClipConfig(cache_dir=str(cache_dir))
    encoder = CLIPTextEmbedder(config, device=device)
    return TaskTextEmbedder(encoder)
