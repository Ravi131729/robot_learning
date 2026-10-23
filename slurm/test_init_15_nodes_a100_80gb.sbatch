#!/bin/bash
#SBATCH --job-name=robot-train-15nodes
#SBATCH --nodes=15
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a100:2
#SBATCH --constraint=gpu_type_a100_80gb
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

module load anaconda3
source activate robot_policy
cd "$HOME/robot_learning"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=online
export WANDB_RUN_GROUP="palmetto-${SLURM_JOB_ID}"
unset WANDB_RUN_ID
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Runs 15 independent two-GPU training runs, each with its own W&B run.
# Stops at the configured step count or the 72-hour Slurm limit.
# Uses existing W&B credentials; checkpoint writes are disabled in this job.
# W&B keeps its normal local logging files; Slurm output is discarded.

srun --kill-on-bad-exit=1 python -u -c '
import os
import sys
import train

# Disable both best and final checkpoint writes without changing train.py.
train.save_checkpoint = lambda *args, **kwargs: None
sys.argv.extend([
    "--wandb-run-name",
    "a100-80gb-job-{}-rank-{}".format(
        os.environ["SLURM_JOB_ID"], os.environ["SLURM_PROCID"]
    ),
])

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
  --timing \
  --memory \
  --wandb \
  --wandb-project robot-policy-ablations \
  --wandb-mode online
