"""ABC camera image preprocessing for DINO-style vision encoders."""

import cv2
import numpy as np


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def to_float_rgb(image):
    """Convert an HWC RGB image to float32 values in `[0, 1]`."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("image must have shape (H, W, 3)")

    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    if np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32)
        if not np.isfinite(image).all() or image.min() < 0.0 or image.max() > 1.0:
            raise ValueError("floating-point images must be finite and in [0, 1]")
        return image
    raise TypeError("images must be uint8 or floating point")


def resize_with_pad(image, target_size=224):
    """Resize an HWC RGB image while preserving aspect ratio and zero-pad."""
    image = to_float_rgb(image)
    height, width, _ = image.shape
    if (height, width) == (target_size, target_size):
        return image

    scale = max(width / target_size, height / target_size)
    new_width = max(1, round(width / scale))
    new_height = max(1, round(height / scale))
    interpolation = cv2.INTER_AREA if scale > 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    padded = np.zeros((target_size, target_size, 3), dtype=np.float32)
    top = (target_size - new_height) // 2
    left = (target_size - new_width) // 2
    padded[top : top + new_height, left : left + new_width] = resized
    return padded


def preprocess_image(image, target_size=224):
    """Return one ImageNet-normalized image in CHW float32 format."""
    image = resize_with_pad(image, target_size=target_size)
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(image, (2, 0, 1)).astype(np.float32)


def preprocess_camera_batch(images, target_size=224):
    """Preprocess a batch of HWC images into `(B, 3, S, S)`."""
    images = np.asarray(images)
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("images must have shape (B, H, W, 3)")
    return np.stack(
        [preprocess_image(image, target_size=target_size) for image in images],
        axis=0,
    )


def preprocess_camera_dict(images, camera_order=("top", "left", "right"), target_size=224):
    """Preprocess camera batches into `(B, C, 3, S, S)` order."""
    missing = [camera for camera in camera_order if camera not in images]
    if missing:
        raise KeyError(f"missing camera images: {missing}")

    processed = [
        preprocess_camera_batch(images[camera], target_size=target_size)
        for camera in camera_order
    ]
    return np.stack(processed, axis=1).astype(np.float32)
