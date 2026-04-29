#!/bin/bash
#SBATCH --job-name=act_aloha
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH --ntasks 1           # number of tasks
#SBATCH --cpus-per-task 16   # number of cpu cores per task
#SBATCH --time 1:00:00       # walltime
#SBATCH --mem 64gb           # amount of memory per CPU core (Memory per Task / Cores per Task)
#SBATCH --nodes 1            # number of nodes
#SBATCH --gpus-per-task l40s:1 # gpu model and amount requested

# Created with the RCD Docs Job Builder
#
# Visit the following link to edit this job:
# https://docs.rcd.clemson.edu/palmetto/job_management/job_builder/?num_cores=16&num_mem=64gb&use_gpus=yes&gpu_model=l40s

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
# -------------------------------
# Load environment
# -------------------------------

module load anaconda3

source activate lerobot

# -------------------------------
# Fix common cluster issues
# -------------------------------

# Use conda libs (fixes GLIBCXX issue)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Headless MuJoCo rendering (fixes your crash)
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# GPU selection
export CUDA_VISIBLE_DEVICES=0

# Optional: avoid torchcodec issues
export LEROBOT_VIDEO_BACKEND=opencv

# -------------------------------
# Config
# -------------------------------
DEVICE=cuda
OUTPUT_DIR=outputs/train/act_aloha_insertion_local
JOB_NAME=act_aloha_insertion_local

# 🔥 tuned batch size (adjust if needed)
BATCH_SIZE=64
STEPS=100000

echo "Starting job on $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Batch size: $BATCH_SIZE"

# -------------------------------
# Run training
# -------------------------------
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
  --eval_freq=5000 \              # less frequent eval (important)
  --eval.batch_size=1 \
  --save_freq=1000 \
  --save_checkpoint=true \
  --log_freq=10 \
  --wandb.enable=True \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}"

echo "Job finished"