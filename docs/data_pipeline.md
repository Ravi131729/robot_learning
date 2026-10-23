# Data pipeline

## ABC episode layout

Each episode is a directory containing data such as:

```text
episode_directory/
├── states_actions.bin
├── combined_camera-images-rgb.mp4
├── episode_metadata.json
├── randomization.json
├── subtasks.json          # optional
└── operator.json          # optional
```

`states_actions.bin` stores floating-point state/action rows. Metadata identifies cameras, task names, instructions, and optional conditioning information. The combined MP4 contains synchronized camera images.

## Window sampling

The ABC `EpisodeDataset` maps a global index to an episode and starting frame. A window contains the state at the first frame and an action chunk of length 30. The video frame at the starting index supplies the camera observation.

## Normalization and image processing

State and action statistics are loaded from `norm_stats.json`. ABC normalization is applied before the data reaches the policy. The model therefore trains on normalized state and action values.

Images are resized and padded to 224×224 and use the ImageNet-style normalization expected by the current DINOv3 path. Training mode also applies random rotation/crop and color augmentation. Validation uses deterministic preprocessing.

## Prompt construction

The loader creates a prompt from the recorded instruction or task name. Optional subtask and operator metadata can contribute to the prompt. The CLIP text embedder converts the prompt into a 512-dimensional task vector.

## Worker loading

`data/abc_official_loader.py` constructs ABC's worker-based PyTorch `DataLoader` with pinned memory, persistent workers, ABC collation, deterministic loader seeding, and TorchCodec video decoding.

`num_workers` controls CPU loading and decoding. It does not increase JAX model compute. Once loader timing is much lower than DINO and policy time, adding workers will have little effect.

## Batch boundaries

```text
ABC batch dictionary
    ↓ data/abc_official_batch.py
host feature batch or PolicyBatch
    ↓
JAX policy update
```

The synchronous path creates JAX arrays immediately. The asynchronous path produces host NumPy feature batches in a separate process and the policy process places them on its JAX device.
