"""Hugging Face DINOv3 ViT-B/16 adapter."""

import numpy as np

from configs.robot_policy_config import DINO_DIM, NUM_VISION_TOKENS


class HFDinoV3Encoder:
    """Expose Hugging Face DINOv3 tokens in the ABC policy format.

    Hugging Face DINOv3 ViT-B/16 returns 201 tokens: one CLS token, four
    register tokens, and 196 patch tokens. ABC policy features retain the CLS
    token and patch tokens, producing the required 197-token sequence.
    """

    def __init__(self, model, device="cpu"):
        self.model = model.to(device).eval()
        self.device = device
        self.num_register_tokens = int(
            getattr(model.config, "num_register_tokens", 0)
        )

        if int(model.config.hidden_size) != DINO_DIM:
            raise ValueError(
                f"expected DINO hidden size {DINO_DIM}, "
                f"got {model.config.hidden_size}"
            )

    @classmethod
    def from_pretrained(cls, model_dir, device="cpu"):
        """Load a local Hugging Face DINOv3 checkpoint."""
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "HFDinoV3Encoder requires the transformers package"
            ) from exc

        model = AutoModel.from_pretrained(
            str(model_dir),
            local_files_only=True,
        )
        return cls(model, device=device)

    def encode_image_tokens(self, images):
        """Encode normalized `(B, 3, 224, 224)` tensors into `(B, 197, 768)`."""
        try:
            import torch
        except ImportError as exc:
            raise ImportError("DINO token encoding requires PyTorch") from exc

        if not isinstance(images, torch.Tensor):
            raise TypeError("images must be a PyTorch tensor")
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 224, 224):
            raise ValueError(
                "images must have shape (B, 3, 224, 224); "
                f"got {tuple(images.shape)}"
            )

        with torch.inference_mode():
            outputs = self.model(
                pixel_values=images.to(self.device),
            )
            hidden = outputs.last_hidden_state

        expected_input_tokens = NUM_VISION_TOKENS + self.num_register_tokens
        if hidden.shape[1] != expected_input_tokens:
            raise ValueError(
                f"expected {expected_input_tokens} DINO tokens before "
                f"register removal, got {hidden.shape[1]}"
            )

        cls_token = hidden[:, :1]
        patch_start = 1 + self.num_register_tokens
        patch_tokens = hidden[:, patch_start:]
        tokens = torch.cat([cls_token, patch_tokens], dim=1)
        if tuple(tokens.shape[1:]) != (NUM_VISION_TOKENS, DINO_DIM):
            raise ValueError(
                f"unexpected DINO token shape after register removal: "
                f"{tuple(tokens.shape)}"
            )

        return tokens


def encode_preprocessed_camera_images(encoder, camera_images):
    """Encode `(B, C, 3, 224, 224)` images into policy token order."""
    from data.dino_tokens import encode_dino_tokens

    return encode_dino_tokens(encoder, np.asarray(camera_images))
