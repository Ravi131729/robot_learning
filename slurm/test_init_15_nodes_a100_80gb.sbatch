#!/bin/bash
#SBATCH --job-name=robot-init-15nodes
#SBATCH --nodes=15
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a100:2
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

module load anaconda3
source activate robot_policy
cd "$HOME/robot_learning"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Runs the existing two-GPU setup independently on all 15 nodes.
# Checks model/optimizer initialization and the first encoded batch.
# Exits before the first update; no checkpoints, W&B files, or logs.
# This checks each node's setup, not inter-node communication.
# sbatch --wait <this-file> returns success only if every task succeeds.

srun --kill-on-bad-exit=1 python -u -c '
import jax
import train

def init_only_step_factory(*factory_args, **factory_kwargs):
    def init_only_step(*args, **kwargs):
        jax.block_until_ready((args, kwargs))
        raise SystemExit(0)
    return init_only_step

train.make_batch_train_step = init_only_step_factory
train.main()
' \
  --data-root "$HOME/robot_learning/cache" \
  --dino-model "$HOME/robot_learning/cache/dinov3-vitb16-pretrain-lvd1689m" \
  --clip-cache "$HOME/robot_learning/cache/clip" \
  --preset medium \
  --batch-size 16 \
  --num-workers 8 \
  --steps 100000 \
  --eval-every 0 \
  --val-steps 1 \
  --pipeline-mode async \
  --encoder-device cuda:0 \
  --policy-device cuda:1 \
  --prefetch-batches 4 \
  --wandb-mode disabled
