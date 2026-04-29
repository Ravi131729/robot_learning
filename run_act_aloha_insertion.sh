#!/bin/bash
set -euo pipefail

# --- Configuration ---
GPU_ID=0
DATASET_REPO_ID="lerobot/aloha_sim_insertion_human"
OUTPUT_DIR="outputs/train/act_aloha_insertion"
WANDB_ENABLE=false   # set to true to enable Weights & Biases logging
NUM_EPOCHS=100
BATCH_SIZE=8
SEED=1000

export CUDA_VISIBLE_DEVICES=${GPU_ID}

echo "Running ACT policy training on ALOHA insertion (GPU ${GPU_ID})"

python lerobot/scripts/train.py \
    policy=act \
    env=aloha \
    env.task=AlohaInsertion-v0 \
    dataset_repo_id=${DATASET_REPO_ID} \
    training.num_epochs=${NUM_EPOCHS} \
    training.batch_size=${BATCH_SIZE} \
    training.seed=${SEED} \
    hydra.run.dir=${OUTPUT_DIR} \
    wandb.enable=${WANDB_ENABLE}

# --- Evaluation (uncomment to run eval after training) ---
CHECKPOINT="${OUTPUT_DIR}/checkpoints/last/pretrained_model"
python lerobot/scripts/eval.py \
    -p ${CHECKPOINT} \
    eval.n_episodes=10000 \
    eval.batch_size=10
