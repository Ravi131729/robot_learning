# Training

## Important options

| Option | Purpose |
|---|---|
| `--data-root` | Dataset and asset cache root |
| `--preset` | `debug`, `small`, `medium`, or `full` |
| `--batch-size` | Samples per policy update |
| `--num-workers` | ABC loader worker count |
| `--steps` | Number of optimizer updates |
| `--eval-every` | Validation interval; `0` disables periodic validation |
| `--val-steps` | Number of validation batches |
| `--sample-steps` | Flow sampler integration steps |
| `--learning-rate` | AdamW learning rate |
| `--weight-decay` | AdamW weight decay |
| `--seed` | Random seed |
| `--timing` | Print pipeline timings |
| `--memory` | Print memory snapshots |
| `--wandb` | Enable W&B logging |
| `--checkpoint-dir` | Checkpoint output directory |

## Single-GPU training

```bash
python train.py \
  --preset small \
  --batch-size 16 \
  --num-workers 8 \
  --steps 1000 \
  --eval-every 200 \
  --val-steps 10 \
  --timing \
  --memory
```

## Two-GPU asynchronous training

```bash
python train.py \
  --preset medium \
  --batch-size 16 \
  --num-workers 8 \
  --steps 1000 \
  --pipeline-mode async \
  --encoder-device cuda:0 \
  --policy-device cuda:1 \
  --prefetch-batches 4 \
  --eval-every 200 \
  --val-steps 10 \
  --timing
```

## Validation and checkpoints

Periodic validation runs every `--eval-every` steps. The final validation always runs after training. The best checkpoint is saved when validation loss improves; the latest checkpoint is saved at the end.

Checkpoints are written as:

```text
checkpoints/<preset>_latest.pkl
checkpoints/<preset>_best.pkl
```

The payload contains the training step, policy parameters, optimizer state, loss history, and command-line arguments.

## W&B

```bash
wandb login
python train.py --preset medium --batch-size 16 --steps 1000 \
  --wandb \
  --wandb-project robot-policy-ablations \
  --wandb-run-name medium-bs16
```

Training loss, validation loss, timing, throughput, parameter counts, memory, and sampled-action statistics are logged when W&B is enabled.

## Reproducibility

Keep the dataset split, seed, number of steps, validation settings, normalization files, and backbone weights fixed when comparing ablations.
