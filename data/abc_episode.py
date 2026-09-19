"""Read-only access to one converted ABC simulation episode."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from configs.robot_policy_config import ACTION_DIM, STATE_DIM


@dataclass(frozen=True)
class ABCActionWindow:
    """A state observation paired with a contiguous demonstrated action chunk."""

    state: np.ndarray
    actions: np.ndarray
    start: int


@dataclass
class ABCEpisode:
    """Metadata and state/action access for one converted ABC episode.

    The ABC converted format stores one float64 row per timestep in
    ``states_actions.bin``. Each row contains 14 state values followed by
    14 action values. The memory map is read-only and opened lazily.
    """

    path: Path
    metadata: dict
    state_dim: int = STATE_DIM
    action_dim: int = ACTION_DIM

    def __post_init__(self):
        self.path = Path(self.path)
        self._state_actions = None
        self._validate_files_and_metadata()

    @classmethod
    def open(cls, episode_dir, state_dim=STATE_DIM, action_dim=ACTION_DIM):
        """Open an episode directory without loading its arrays into RAM."""
        episode_dir = Path(episode_dir)
        metadata_path = episode_dir / "episode_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"missing episode metadata: {metadata_path}")

        metadata = json.loads(metadata_path.read_text())
        return cls(
            path=episode_dir,
            metadata=metadata,
            state_dim=state_dim,
            action_dim=action_dim,
        )

    @property
    def episode_id(self):
        return self.metadata.get("episode_id", self.path.name)

    @property
    def task_name(self):
        return self.metadata.get("task_name")

    @property
    def instruction(self):
        return self.metadata.get("instruction", self.task_name)

    @property
    def num_steps(self):
        return int(self.metadata["num_steps"])

    @property
    def fps(self):
        return float(self.metadata.get("fps", 30.0))

    @property
    def cameras(self):
        return tuple(self.metadata.get("cameras", ("top", "left", "right")))

    @property
    def image_size(self):
        return (
            int(self.metadata["image_width"]),
            int(self.metadata["image_height"]),
        )

    @property
    def row_width(self):
        return self.state_dim + self.action_dim

    @property
    def state_actions_path(self):
        return self.path / "states_actions.bin"

    @property
    def video_path(self):
        return self.path / "combined_camera-images-rgb.mp4"

    @property
    def state_actions(self):
        """Return the read-only memory-mapped `(T, 28)` state/action array."""
        if self._state_actions is None:
            self._state_actions = np.memmap(
                self.state_actions_path,
                mode="r",
                dtype=np.float64,
                shape=(self.num_steps, self.row_width),
            )
        return self._state_actions

    def read_rows(self, start=0, end=None):
        """Read state/action rows as a float32 NumPy array."""
        start, end = self._validate_range(start, end)
        return np.asarray(self.state_actions[start:end], dtype=np.float32)

    def read_state(self, timestep):
        """Read one 14-dimensional state vector."""
        timestep = self._validate_timestep(timestep)
        return np.asarray(
            self.state_actions[timestep, : self.state_dim],
            dtype=np.float32,
        )

    def read_action(self, timestep):
        """Read one 14-dimensional action vector."""
        timestep = self._validate_timestep(timestep)
        return np.asarray(
            self.state_actions[timestep, self.state_dim :],
            dtype=np.float32,
        )

    def read_window(self, start, horizon):
        """Read the state at `start` and the next `horizon` actions."""
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        self._validate_range(start, start + horizon)

        rows = self.state_actions[start : start + horizon]
        return ABCActionWindow(
            state=np.asarray(rows[0, : self.state_dim], dtype=np.float32),
            actions=np.asarray(rows[:, self.state_dim :], dtype=np.float32),
            start=start,
        )

    def valid_window_starts(self, horizon):
        """Return the number of valid action-window start indices."""
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        return max(0, self.num_steps - horizon + 1)

    def _validate_files_and_metadata(self):
        if not self.path.is_dir():
            raise NotADirectoryError(f"episode path is not a directory: {self.path}")
        if not self.state_actions_path.exists():
            raise FileNotFoundError(f"missing state/action file: {self.state_actions_path}")
        if self.num_steps < 1:
            raise ValueError(f"episode has invalid num_steps: {self.num_steps}")

        expected_bytes = self.num_steps * self.row_width * np.dtype(np.float64).itemsize
        actual_bytes = self.state_actions_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"state/action size mismatch for {self.path.name}: "
                f"expected {expected_bytes} bytes, found {actual_bytes}"
            )

    def _validate_timestep(self, timestep):
        timestep = int(timestep)
        if not 0 <= timestep < self.num_steps:
            raise IndexError(
                f"timestep {timestep} outside [0, {self.num_steps})"
            )
        return timestep

    def _validate_range(self, start, end):
        start = int(start)
        end = self.num_steps if end is None else int(end)
        if not 0 <= start <= end <= self.num_steps:
            raise IndexError(
                f"range [{start}, {end}) outside [0, {self.num_steps})"
            )
        return start, end


def list_episodes(cache_root, split="train_sim"):
    """List valid ABC episode directories in a cache split."""
    split_root = Path(cache_root) / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"ABC split directory not found: {split_root}")

    return tuple(
        sorted(
            episode_dir
            for episode_dir in split_root.glob("episode_*")
            if episode_dir.is_dir()
        )
    )
