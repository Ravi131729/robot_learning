"""Read synchronized camera frames from converted ABC episode videos.

The preferred backend follows ABC's ``abc_minimal.dataloader.decode_frame``
implementation and uses an indexed ``torchcodec.VideoDecoder``. PyAV is kept
as a fallback so dataset inspection still works before torchcodec is
installed. The torchcodec behavior is copied from Amazon FAR ABC:
https://github.com/amazon-far/abc
"""

from dataclasses import dataclass, field
from pathlib import Path
import warnings

import numpy as np


@dataclass
class ABCVideoReader:
    """Read frames from ABC's vertically stacked RGB camera video."""

    video_path: Path
    cameras: tuple[str, ...]
    frame_width: int
    frame_height: int
    expected_frames: int | None = None
    backend: str = "torchcodec"
    _decoder: object = field(default=None, init=False, repr=False)

    @classmethod
    def from_episode(cls, episode):
        """Create a reader from an :class:`data.abc_episode.ABCEpisode`."""
        return cls(
            video_path=episode.video_path,
            cameras=episode.cameras,
            frame_width=episode.image_size[0],
            frame_height=episode.image_size[1],
            expected_frames=episode.num_steps,
        )

    @property
    def stacked_height(self):
        return self.frame_height * len(self.cameras)

    def read_stacked_frame(self, index):
        """Read one RGB frame with shape `(num_cameras*H, W, 3)`."""
        index = self._validate_index(index)
        if not self.video_path.exists():
            raise FileNotFoundError(f"missing episode video: {self.video_path}")

        if self.backend == "torchcodec":
            try:
                decoder = self._get_torchcodec_decoder()
            except (ImportError, OSError, RuntimeError) as exc:
                warnings.warn(
                    "torchcodec is installed but unavailable; falling back to "
                    f"PyAV ({type(exc).__name__}: {exc})",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.backend = "pyav"
            else:
                frame = decoder[index]
                # torchcodec returns (C, H, W) uint8 tensors.
                image = frame.permute(1, 2, 0).contiguous().cpu().numpy()
                return self._validate_frame(image)

        return self._read_pyav_frame(index)

    def read_camera_frames(self, index):
        """Read one timestep as `{camera_name: RGB image}`."""
        stacked = self.read_stacked_frame(index)
        return {
            camera: stacked[camera_index * self.frame_height : (camera_index + 1) * self.frame_height]
            for camera_index, camera in enumerate(self.cameras)
        }

    def read_frames(self, indices):
        """Read multiple frames in one sequential decode pass.

        Returned frames preserve the order of `indices` and have shape
        `(N, num_cameras*H, W, 3)`.
        """
        requested = [self._validate_index(index) for index in indices]
        if not requested:
            return np.empty(
                (0, self.stacked_height, self.frame_width, 3),
                dtype=np.uint8,
            )

        return np.stack([self.read_stacked_frame(index) for index in requested], axis=0)

    def split_cameras(self, stacked_frames):
        """Split stacked frames into a camera-first array `(N, C, H, W, 3)`."""
        frames = np.asarray(stacked_frames)
        if frames.ndim == 3:
            frames = frames[None]
        if frames.ndim != 4:
            raise ValueError("stacked_frames must have shape (N, H_total, W, 3)")
        if frames.shape[1:] != (
            self.stacked_height,
            self.frame_width,
            3,
        ):
            raise ValueError(
                f"unexpected stacked frame shape: {frames.shape}; "
                f"expected (N, {self.stacked_height}, {self.frame_width}, 3)"
            )

        return frames.reshape(
            frames.shape[0],
            len(self.cameras),
            self.frame_height,
            self.frame_width,
            3,
        )

    def _validate_index(self, index):
        index = int(index)
        if index < 0:
            raise IndexError("frame index must be non-negative")
        if self.expected_frames is not None and index >= self.expected_frames:
            raise IndexError(
                f"frame index {index} outside [0, {self.expected_frames})"
            )
        return index

    def _get_torchcodec_decoder(self):
        if self._decoder is None:
            try:
                from torchcodec.decoders import VideoDecoder
            except ImportError as exc:
                raise ImportError(
                    "torchcodec is required for the ABC video backend; "
                    "install it with `python -m pip install torchcodec`"
                ) from exc
            self._decoder = VideoDecoder(str(self.video_path))
        return self._decoder

    def _read_pyav_frame(self, index):
        try:
            import av
        except ImportError as exc:
            raise ImportError(
                "neither torchcodec nor PyAV is installed for video decoding"
            ) from exc

        with av.open(str(self.video_path)) as container:
            stream = container.streams.video[0]
            for frame_index, frame in enumerate(container.decode(stream)):
                if frame_index == index:
                    image = frame.to_ndarray(format="rgb24")
                    return self._validate_frame(image)

        raise IndexError(f"video contains fewer than {index + 1} frames")

    def _validate_frame(self, image):
        expected_shape = (self.stacked_height, self.frame_width, 3)
        if image.shape != expected_shape:
            raise ValueError(
                f"unexpected decoded frame shape: {image.shape}; "
                f"expected {expected_shape}"
            )
        return image
