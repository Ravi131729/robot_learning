#!/bin/bash
#SBATCH --job-name=robot-async-test
#SBATCH --gpus-per-node=a100:2
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module load anaconda3
source activate robot_policy
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "Host: $(hostname)"
nvidia-smi

srun python train.py \
  --data-root "$HOME/robot_learning/cache" \
  --dino-model "$HOME/robot_learning/cache/dinov3-vitb16-pretrain-lvd1689m" \
  --clip-cache "$HOME/robot_learning/cache/clip" \
  --preset debug \
  --batch-size 4 \
  --num-workers 8 \
  --steps 10 \
  --eval-every 0 \
  --val-steps 2 \
  --pipeline-mode async \
  --encoder-device cuda:0 \
  --policy-device cuda:1 \
  --prefetch-batches 4 \
  --timing \
  --memory \
  --wandb-mode offline
