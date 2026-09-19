#!/bin/bash
#SBATCH --job-name=robot-ablation
#SBATCH --gpus-per-node=a100:2
#SBATCH --constraint=gpu_a100_80gb
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-5%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

module load anaconda3
source activate robot_policy
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p logs

PRESETS=(medium medium medium full full full)
BATCHES=(16 32 64 16 32 64)

PRESET="${PRESETS[$SLURM_ARRAY_TASK_ID]}"
BATCH="${BATCHES[$SLURM_ARRAY_TASK_ID]}"
RUN_NAME="a100-80gb-depth-${PRESET}-bs${BATCH}"

echo "Running ${RUN_NAME} on ${CUDA_VISIBLE_DEVICES:-scheduler-assigned GPU}"
nvidia-smi

srun python train.py \
  --data-root "$HOME/robot_learning/cache" \
  --dino-model "$HOME/robot_learning/cache/dinov3-vitb16-pretrain-lvd1689m" \
  --clip-cache "$HOME/robot_learning/cache/clip" \
  --preset "$PRESET" \
  --batch-size "$BATCH" \
  --num-workers 8 \
  --pipeline-mode async \
  --encoder-device cuda:0 \
  --policy-device cuda:1 \
  --prefetch-batches 4 \
  --steps 5000 \
  --eval-every 500 \
  --val-steps 20 \
  --timing \
  --memory \
  --wandb \
  --wandb-project robot-policy-ablations \
  --wandb-run-name "$RUN_NAME"
