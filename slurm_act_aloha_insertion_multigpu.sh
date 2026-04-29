#!/bin/bash
#SBATCH --job-name=act_aloha_insertion
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

set -euo pipefail

# --- Configuration ---
NUM_GPUS=4
DATASET_REPO_ID="lerobot/aloha_sim_insertion_human"
OUTPUT_DIR="outputs/train/act_aloha_insertion_multigpu"
NUM_EPOCHS=100
BATCH_SIZE=8         # per-GPU batch size
SEED=1000
WANDB_PROJECT="lerobot_act"
WANDB_RUN_NAME="act_aloha_insertion_${SLURM_JOB_ID}"

# --- Environment setup ---
# Activate your conda/venv environment here, e.g.:
# source activate lerobot
# or: source /path/to/venv/bin/activate

export WANDB_PROJECT=${WANDB_PROJECT}
export WANDB_RUN_ID=${WANDB_RUN_NAME}

mkdir -p logs

echo "Job ID:      ${SLURM_JOB_ID}"
echo "Node:        ${SLURMD_NODENAME}"
echo "GPUs:        ${NUM_GPUS}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "W&B project: ${WANDB_PROJECT}"
echo "W&B run:     ${WANDB_RUN_NAME}"

# --- Write accelerate config inline ---
ACCEL_CONFIG=$(mktemp /tmp/accelerate_config_XXXXXX.yaml)
cat > "${ACCEL_CONFIG}" <<EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_processes: ${NUM_GPUS}
num_machines: 1
mixed_precision: bf16
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
downcast_bf16: false
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

echo "Accelerate config written to ${ACCEL_CONFIG}"

# --- Launch training ---
accelerate launch \
    --config_file "${ACCEL_CONFIG}" \
    --num_processes ${NUM_GPUS} \
    lerobot/scripts/train.py \
        policy=act \
        env=aloha \
        env.task=AlohaInsertion-v0 \
        dataset_repo_id=${DATASET_REPO_ID} \
        training.num_epochs=${NUM_EPOCHS} \
        training.batch_size=${BATCH_SIZE} \
        training.seed=${SEED} \
        hydra.run.dir=${OUTPUT_DIR} \
        wandb.enable=true \
        wandb.project=${WANDB_PROJECT} \
        wandb.run_name=${WANDB_RUN_NAME}

rm -f "${ACCEL_CONFIG}"
