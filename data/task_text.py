"""Task-prompt normalization and 512D text-embedding adapters."""

import numpy as np

from configs.robot_policy_config import TASK_DIM


def task_name_to_prompt(task_name):
    """Convert ABC task identifiers into human-readable prompts."""
    return " ".join(
        str(task_name).replace("-", " ").replace("_", " ").split()
    )


def resolve_prompt(task_name=None, instruction=None):
    """Prefer the recorded instruction, falling back to the task name."""
    if isinstance(instruction, str) and instruction.strip():
        return instruction.strip()
    if task_name is None:
        raise ValueError("task_name or instruction is required")
    return task_name_to_prompt(task_name)


class TaskTextEmbedder:
    """Adapt an ABC-compatible text encoder to normalized NumPy vectors.

    The encoder must either expose ``encode(list[str])`` or be callable with a
    list of prompt strings. Its output must have shape `(B, 512)`. ABC's
    ``CLIPTextEmbedder`` satisfies this interface directly.
    """

    def __init__(self, encoder):
        self.encoder = encoder

    def encode_prompts(self, prompts):
        prompts = list(prompts)
        if hasattr(self.encoder, "encode"):
            embeddings = self.encoder.encode(prompts)
        else:
            embeddings = self.encoder(prompts)

        if hasattr(embeddings, "detach"):
            embeddings = embeddings.detach().cpu().numpy()
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.shape != (len(prompts), TASK_DIM):
            raise ValueError(
                f"task encoder must return {(len(prompts), TASK_DIM)}, "
                f"got {embeddings.shape}"
            )
        if not np.isfinite(embeddings).all():
            raise ValueError("task embeddings contain non-finite values")
        return embeddings

    def encode_task_names(self, task_names):
        return self.encode_prompts(
            [task_name_to_prompt(name) for name in task_names]
        )

    def encode_samples(self, samples):
        return self.encode_prompts(
            [
                resolve_prompt(sample.task_name, sample.instruction)
                for sample in samples
            ]
        )
