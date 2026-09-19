"""Asynchronous frozen-feature producer for the two-GPU training pipeline.

The producer owns the ABC loader and frozen Torch encoders. It emits fresh
features for every batch, so training-time image augmentation is preserved.
The policy process consumes host NumPy batches and places them on its JAX
device. Training-loop integration is intentionally kept separate until this
producer/queue boundary has been validated.
"""

from dataclasses import dataclass
import multiprocessing as mp
import queue
import time
import traceback

from data.abc_official_batch import encode_official_batch_host
from data.abc_official_loader import create_abc_loader
from data.abc_clip import load_abc_clip_text_embedder
from data.hf_dino import HFDinoV3Encoder


_STOP = "__robot_policy_stop__"


@dataclass(frozen=True)
class AsyncFeatureConfig:
    data_root: str
    dino_model: str
    clip_cache: str
    split: str = "train_sim"
    batch_size: int = 1
    num_workers: int = 4
    encoder_device: str = "cuda:0"
    prefetch_batches: int = 4
    abc_root: str | None = None
    seed: int = 0


def _producer_main(output_queue, error_queue, config):
    """Run in a spawned process and continuously produce host feature batches."""
    try:
        loader = create_abc_loader(
            config.data_root,
            split=config.split,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            abc_root=config.abc_root,
            seed=config.seed,
            train=True,
        )
        dino = HFDinoV3Encoder.from_pretrained(
            config.dino_model,
            device=config.encoder_device,
        )
        clip = load_abc_clip_text_embedder(
            abc_root=config.abc_root,
            cache_dir=config.clip_cache,
            device=config.encoder_device,
        )

        iterator = iter(loader)
        while True:
            try:
                raw_batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                raw_batch = next(iterator)
            start = time.perf_counter()
            host_batch, timings = encode_official_batch_host(
                raw_batch, dino, clip, with_timings=True
            )
            timings["producer total"] = time.perf_counter() - start
            output_queue.put({"batch": host_batch, "timings": timings})
    except BaseException:
        error_queue.put(traceback.format_exc())


class AsyncFeatureQueue:
    """Bounded multiprocessing queue backed by a frozen-feature producer."""

    def __init__(self, config: AsyncFeatureConfig):
        if config.prefetch_batches < 1:
            raise ValueError("prefetch_batches must be at least 1")
        self.config = config
        self.context = mp.get_context("spawn")
        self.output_queue = self.context.Queue(maxsize=config.prefetch_batches)
        self.error_queue = self.context.Queue(maxsize=1)
        self.process = self.context.Process(
            target=_producer_main,
            args=(self.output_queue, self.error_queue, config),
            name="robot-policy-feature-producer",
        )

    def start(self):
        self.process.start()
        return self

    def get(self, timeout=None):
        """Return the next host batch, raising a producer exception if needed."""
        try:
            error = self.error_queue.get_nowait()
        except queue.Empty:
            error = None
        if error is not None:
            raise RuntimeError("asynchronous feature producer failed:\n" + error)
        try:
            item = self.output_queue.get(timeout=timeout)
        except queue.Empty:
            try:
                error = self.error_queue.get_nowait()
            except queue.Empty:
                raise
            raise RuntimeError("asynchronous feature producer failed:\n" + error)
        return item

    def close(self):
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=10)
        if self.process.is_alive():
            self.process.kill()
            self.process.join()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback_value):
        self.close()
