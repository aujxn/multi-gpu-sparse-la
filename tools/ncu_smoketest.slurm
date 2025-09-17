#!/bin/bash
#SBATCH --job-name=ncu_smoketest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --partition=short
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module load cuda
# If Nsight Compute is a separate module on your cluster, uncomment:
# module load nsight-compute

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs ncu_reports

# Ensure CUPTI on LD_LIBRARY_PATH
if command -v nvcc >/dev/null 2>&1; then
  CUDA_BIN=$(command -v nvcc)
  CUDA_HOME=$(cd "$(dirname "$CUDA_BIN")/.." && pwd)
  export CUDA_HOME
  if [ -d "$CUDA_HOME/extras/CUPTI/lib64" ]; then
    export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
  fi
fi

echo "Using ncu: $(command -v ncu)" >&2
ncu --version || true

# Build a minimal CUDA binary in node-local tmp
TMPDIR=/tmp/ncu_smoke_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
cp tools/cuda_smoketest.cu "$TMPDIR/"
pushd "$TMPDIR" >/dev/null
nvcc -O2 -std=c++17 -arch=compute_80 -code=sm_80,compute_80 cuda_smoketest.cu -o cuda_smoketest
popd >/dev/null

echo "Running Nsight Compute smoketest under srun..." >&2
srun -N1 -n1 ncu --set basic --replay-mode application \
  -o "$SLURM_SUBMIT_DIR/ncu_reports/ncu_smoke_${SLURM_JOB_ID}" "$TMPDIR/cuda_smoketest"

echo "Smoketest report: $SLURM_SUBMIT_DIR/ncu_reports/ncu_smoke_${SLURM_JOB_ID}.ncu-rep" >&2
