"""Loader for ABC's official OpenAI CLIP text embedder."""

from pathlib import Path
import sys

from data.task_text import TaskTextEmbedder


def load_abc_clip_text_embedder(
    abc_root="/home/ravi/abc",
    cache_dir="/home/ravi/robot_learning/cache/clip",
    device="cpu",
):
    """Load ABC's CLIP ViT-B/32 text encoder from local assets."""
    abc_root = Path(abc_root)
    if not (abc_root / "abc_minimal").is_dir():
        raise FileNotFoundError(f"ABC source directory not found: {abc_root}")

    abc_root_string = str(abc_root)
    if abc_root_string not in sys.path:
        sys.path.insert(0, abc_root_string)

    try:
        from abc_minimal.config import ClipConfig
        from abc_minimal.dit import CLIPTextEmbedder
    except ImportError as exc:
        raise ImportError(
            "Could not import ABC's CLIP text encoder. "
            "Install the dependencies used by /home/ravi/abc."
        ) from exc

    config = ClipConfig(cache_dir=str(cache_dir))
    encoder = CLIPTextEmbedder(config, device=device)
    return TaskTextEmbedder(encoder)
