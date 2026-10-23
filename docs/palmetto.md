# Clemson Palmetto

## Environment

On a compute node:

```bash
module load anaconda3
source activate robot_policy
```

Use current-user paths explicitly:

```bash
--data-root "$HOME/robot_learning/cache"
--dino-model "$HOME/robot_learning/cache/dinov3-vitb16-pretrain-lvd1689m"
--clip-cache "$HOME/robot_learning/cache/clip"
```

## GPU variants

```bash
# A100 40 GB
#SBATCH --gpus-per-node=a100:2
#SBATCH --constraint=gpu_a100_40gb

# A100 80 GB
#SBATCH --gpus-per-node=a100:2
#SBATCH --constraint=gpu_a100_80gb
```

Always verify the actual allocation with `nvidia-smi`.

## Provided job scripts

```bash
mkdir -p logs
sbatch slurm/test_async_two_gpu.sbatch
sbatch slurm/medium_a100_40gb.sbatch
sbatch slurm/sweep_depth16_32.sbatch
```

The smoke test uses the debug preset and ten steps. The medium job uses depth 16 and batch size 16 on A100 40 GB GPUs. The sweep tests depth 16 and depth 32 at batch sizes 16, 32, and 64 on A100 80 GB GPUs.

## Monitoring

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS
nvidia-smi
```

Create `logs/` before submission because Slurm opens output paths before the script body runs.

## JAX memory behavior

The job scripts set:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

This avoids immediately reserving most visible GPU memory when Torch and JAX share one allocation across two assigned GPUs.
