#!/bin/bash
#SBATCH --job-name=build_cuSPARSE
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=short

set -e

echo "Loading modules..."
module load cuda

echo "Building..."
mkdir -p build && cd build
cmake -DNCCL_ROOT="$NCCL_HOME" ..

cmake --build . --parallel "$SLURM_CPUS_PER_TASK"
cd ..
echo "Build complete."

