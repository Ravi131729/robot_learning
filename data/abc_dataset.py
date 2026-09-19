"""Windowed dataset interface for converted ABC episodes."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from configs.robot_policy_config import ACTION_HORIZON
from data.abc_episode import ABCEpisode, list_episodes
from data.abc_video import ABCVideoReader
from data.normalization import load_abc_norm_stats


@dataclass(frozen=True)
class ABCTrainingSample:
    """One observation and demonstrated action chunk from ABC."""

    episode_id: str
    task_name: str
    instruction: str | None
    start: int
    images: dict[str, np.ndarray]
    state: np.ndarray
    actions: np.ndarray


@dataclass(frozen=True)
class ABCBatch:
    """Collated raw ABC batch before vision/text encoding."""

    images: dict[str, np.ndarray]
    state: np.ndarray
    actions: np.ndarray
    task_names: tuple[str, ...]
    instructions: tuple[str | None, ...]
    episode_ids: tuple[str, ...]
    starts: np.ndarray


class ABCDataset:
    """Sample fixed-length action windows from an ABC cache split.

    Images are returned as raw uint8 RGB arrays with shape `(H, W, 3)`.
    State and action values are normalized with ABC's `norm_stats.json` by
    default. The dataset does not create DINO or task embeddings.
    """

    def __init__(
        self,
        cache_root,
        split="train_sim",
        horizon=ACTION_HORIZON,
        norm_stats_path=None,
        normalize=True,
    ):
        if horizon < 1:
            raise ValueError("horizon must be at least 1")

        self.cache_root = Path(cache_root)
        self.split = split
        self.horizon = int(horizon)
        episode_paths = list_episodes(self.cache_root, split)
        self.episodes = tuple(ABCEpisode.open(path) for path in episode_paths)
        self.video_readers = {
            index: ABCVideoReader.from_episode(episode)
            for index, episode in enumerate(self.episodes)
        }

        self.valid_starts = tuple(
            episode.valid_window_starts(self.horizon)
            for episode in self.episodes
        )
        self.cumulative_starts = np.cumsum(self.valid_starts)
        if not len(self.episodes) or int(self.cumulative_starts[-1]) == 0:
            raise ValueError(f"no usable episodes found in {self.cache_root / split}")

        self.norm_stats = None
        if normalize:
            stats_path = (
                Path(norm_stats_path)
                if norm_stats_path is not None
                else self.cache_root / "norm_stats.json"
            )
            self.norm_stats = load_abc_norm_stats(stats_path)

    def __len__(self):
        return int(self.cumulative_starts[-1])

    def __getitem__(self, index):
        episode_index, start = self._locate(index)
        episode = self.episodes[episode_index]
        window = episode.read_window(start, self.horizon)
        images = self.video_readers[episode_index].read_camera_frames(start)

        state = window.state
        actions = window.actions
        if self.norm_stats is not None:
            state = np.asarray(
                self.norm_stats["state"].normalize(state),
                dtype=np.float32,
            )
            actions = np.asarray(
                self.norm_stats["actions"].normalize(actions),
                dtype=np.float32,
            )

        return ABCTrainingSample(
            episode_id=episode.episode_id,
            task_name=episode.task_name,
            instruction=episode.instruction,
            start=start,
            images=images,
            state=state,
            actions=actions,
        )

    def _locate(self, index):
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(f"sample index {index} outside [0, {len(self)})")

        episode_index = int(np.searchsorted(
            self.cumulative_starts,
            index,
            side="right",
        ))
        previous = (
            0
            if episode_index == 0
            else int(self.cumulative_starts[episode_index - 1])
        )
        return episode_index, index - previous


def collate_abc_samples(samples):
    """Stack samples into an `ABCBatch` without converting image semantics."""
    if not samples:
        raise ValueError("cannot collate an empty sample list")

    camera_names = tuple(samples[0].images)
    if any(tuple(sample.images) != camera_names for sample in samples):
        raise ValueError("all samples must have the same camera names")

    return ABCBatch(
        images={
            camera: np.stack(
                [sample.images[camera] for sample in samples],
                axis=0,
            )
            for camera in camera_names
        },
        state=np.stack([sample.state for sample in samples], axis=0),
        actions=np.stack([sample.actions for sample in samples], axis=0),
        task_names=tuple(sample.task_name for sample in samples),
        instructions=tuple(sample.instruction for sample in samples),
        episode_ids=tuple(sample.episode_id for sample in samples),
        starts=np.asarray([sample.start for sample in samples], dtype=np.int64),
    )
