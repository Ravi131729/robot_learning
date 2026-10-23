# Architecture

## Inputs and outputs

The policy consumes one sampled ABC training window:

| Input | Shape | Meaning |
|---|---:|---|
| Camera images | `B × 3 × 3 × 224 × 224` | Batch, cameras, RGB image |
| DINOv3 tokens | `B × 197 × 3 × 768` | CLS plus 196 patch tokens per camera |
| Robot state | `B × 14` | Normalized state at the window start |
| Task embedding | `B × 512` | CLIP text representation |
| Actions | `B × 30 × 14` | Normalized target action chunk |

The model predicts a flow-matching vector field over the action chunk. Sampling integrates this vector field from noise toward an action prediction.

## High-level flow

```text
ABC episodes
    │
    ├── states_actions.bin ── normalize ── robot state
    ├── MP4 video ─────────── decode ───── camera images
    └── metadata ───────────── prompt ───── task text
                         │
                         ▼
              DINOv3 image tokens + CLIP text embedding
                         │
                         ▼
                 Flow-matching DiT policy
                         │
                         ▼
                    30-step action chunk
```

## Vision path

DINOv3 ViT-B/16 produces 201 tokens: one CLS token, four register tokens, and 196 patch tokens. The adapter removes the four register tokens and keeps 197 tokens:

```text
201 DINO tokens → remove 4 register tokens → 197 policy tokens
```

The policy keeps the camera axis separate until its learned vision conditioning path. Camera-specific pooling and projection convert DINO features into the DiT feature width of 1536.

## Task and state conditioning

The 512-dimensional CLIP task embedding and the 14-dimensional robot state are projected into the policy conditioning representation. The timestep/noise level is embedded separately and used with adaptive layer normalization.

## DiT block

Each block contains three residual stages:

```text
input
  ├── affine-free LayerNorm → AdaLN → self-attention → gate ── residual
  ├── affine-free query LayerNorm → AdaLN
  │       + vision KV LayerNorm → vision cross-attention → gate ── residual
  └── affine-free LayerNorm → AdaLN → MLP → gate ─────────────── residual
```

The default architecture constants are:

```text
DiT width:        1536
Attention heads:  24
MLP width:        6144
Action horizon:   30
State dimension:  14
Action dimension: 14
```

## Presets

| Preset | DiT depth | Intended use |
|---|---:|---|
| `debug` | 1 | Smoke tests and pipeline debugging |
| `small` | 8 | Fast local experiments |
| `medium` | 16 | A100 40 GB experiments |
| `full` | 32 | Large A100 80 GB experiments |

## Training objective and sampling

The flow-matching objective samples a time/noise value, constructs an interpolated noisy action, predicts the vector field, and minimizes the prediction error. At inference time, the sampler repeatedly evaluates the vector field for a configured number of integration steps. Actions remain in normalized ABC coordinates until they are unnormalized for simulator use.
