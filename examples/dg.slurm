#!/bin/bash
#SBATCH --job-name=multi-gpu_dg
#SBATCH --nodes=2
#SBATCH --gres=gpu:l40s:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --partition=short
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00

set -e

module load cuda

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

export NCCL_SOCKET_IFNAME=ens2,ib0
export NCCL_DEBUG=INFO
echo "Running dg example..."
srun -N 2 --gpus-per-node=4 --ntasks-per-node=4 --mpi=pmix \
  ./build/dg_diffusion