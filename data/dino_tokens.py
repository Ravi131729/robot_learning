"""Adapter from a pretrained DINO-style encoder to policy token tensors."""

import numpy as np

from configs.robot_policy_config import DINO_DIM, NUM_CAMERAS, NUM_VISION_TOKENS


def encode_dino_tokens(encoder, camera_images, device=None):
    """Encode `(B, C, 3, H, W)` images into `(B, 197, C, 768)` tokens.

    The encoder must expose ``encode_image_tokens(images)`` and return
    `(B*C, 197, 768)` tokens. This adapter intentionally does not download or
    construct model weights; the caller supplies the pretrained encoder.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("DINO token encoding requires PyTorch") from exc

    camera_images = np.asarray(camera_images)
    if camera_images.ndim != 5:
        raise ValueError("camera_images must have shape (B, C, 3, H, W)")
    batch_size, camera_count, channels, _, _ = camera_images.shape
    if camera_count != NUM_CAMERAS or channels != 3:
        raise ValueError(
            f"expected (B, {NUM_CAMERAS}, 3, H, W), got {camera_images.shape}"
        )

    images = torch.from_numpy(camera_images.reshape(
        batch_size * camera_count,
        channels,
        camera_images.shape[-2],
        camera_images.shape[-1],
    ))
    if device is not None:
        images = images.to(device)

    with torch.no_grad():
        tokens = encoder.encode_image_tokens(images)

    if tokens.ndim != 3 or tokens.shape[1:] != (NUM_VISION_TOKENS, DINO_DIM):
        raise ValueError(
            "encoder must return tokens with shape "
            f"(B*C, {NUM_VISION_TOKENS}, {DINO_DIM}); got {tuple(tokens.shape)}"
        )

    tokens = tokens.reshape(
        batch_size,
        camera_count,
        NUM_VISION_TOKENS,
        DINO_DIM,
    ).permute(0, 2, 1, 3)
    return tokens.detach().float().cpu().numpy()
