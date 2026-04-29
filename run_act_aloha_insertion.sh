#!/usr/bin/env bash
set -euo pipefail

# Local ACT training entry point for the Aloha insertion task.
# Override any of these from your shell before running the script.
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/act_aloha_insertion_local}"
JOB_NAME="${JOB_NAME:-act_aloha_insertion_local}"
BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="${STEPS:-100000}"

lerobot-train \
  --policy.type=act \
  --policy.device="${DEVICE}" \
  --policy.chunk_size=20 \
  --policy.n_action_steps=20 \
  --policy.dim_model=64 \
  --policy.push_to_hub=false \
  --env.type=aloha \
  --env.task=AlohaInsertion-v0 \
  --env.episode_length=400 \
  --dataset.repo_id=lerobot/aloha_sim_insertion_human \
  --dataset.image_transforms.enable=true \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --eval_freq=500 \
  --eval.n_episodes=1 \
  --eval.batch_size=1 \
  --save_freq=500 \
  --save_checkpoint=true \
  --log_freq=10 \
  --wandb.enable=True \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}"