#!/bin/bash
#SBATCH --job-name=bench_hypre_gpu
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

set -euo pipefail

module load cuda
# If your cluster separates Nsight Compute, also load it here, e.g.:
# module load nsight-compute

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
mkdir -p ncu_reports

# Ensure CUPTI and Nsight Compute binaries are on PATH/LD_LIBRARY_PATH
if command -v nvcc >/dev/null 2>&1; then
  CUDA_BIN=$(command -v nvcc)
  CUDA_HOME=$(cd "$(dirname "$CUDA_BIN")/.." && pwd)
  export CUDA_HOME
  if [ -d "$CUDA_HOME/extras/CUPTI/lib64" ]; then
    export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
  fi
fi
which ncu || true
ncu --version || true
echo "LD_LIBRARY_PATH begins with: ${LD_LIBRARY_PATH%%:*}" || true

# Network settings for consistency (not strictly needed here)
export NCCL_SOCKET_IFNAME=ens2,ib0
export NCCL_DEBUG=INFO

# Open MPI + UCX: prefer CUDA-aware transports and GPUDirect if available
#export OMPI_MCA_pml=ucx
#export OMPI_MCA_osc=ucx
# Common UCX transport set for multi-node with GPU buffers
#export UCX_TLS=sm,self,cuda_copy,cuda_ipc,rc_x
# Enable GPUDirect RDMA if nvidia_peermem is present
#export UCX_IB_GPU_DIRECT_RDMA=y
# Memory type cache can help with buffer detection on some stacks
#export UCX_MEMTYPE_CACHE=y

#export HYPRE_SPMV_VENDOR=1

echo "Running Hypre GPU SpMV benchmark..."

# Optional: enable Nsight Compute profiling by setting PROFILE_WITH_NCU=1
# You can also customize the preset via NCU_SET (e.g., speedOfLight, launchStats, full)
# Example:
#   PROFILE_WITH_NCU=1 NCU_SET=speedOfLight sbatch benchmarks/poisson_bench_hypre_gpu.slurm
if [[ "${PROFILE_WITH_NCU:-0}" -eq 1 ]]; then
  mkdir -p ncu_reports
  REQ_SET="${NCU_SET:-speedOfLight}"
  AVAIL_SETS=$(ncu --list-sets 2>/dev/null | sed -n 's/^[[:space:]]*\([[:alnum:]_][[:alnum:]_]*\)[[:space:]]*$/\1/p' || true)
  if ! echo "$AVAIL_SETS" | grep -qx "$REQ_SET"; then
    if echo "$AVAIL_SETS" | grep -qx speedOfLight; then
      REQ_SET="speedOfLight"
    else
      REQ_SET="basic"
    fi
  fi
  echo "Nsight Compute using set: $REQ_SET" >&2
  export NCU_VERBOSE="${NCU_VERBOSE:-2}"
  export NCU_ARGS="--set ${REQ_SET} --replay-mode application --force-overwrite true --kernel-name-base demangled --profile-from-start on --verbose ${NCU_VERBOSE} ${NCU_ARGS_EXTRA:-}"

  if [[ "${PROFILE_ALL_RANKS:-1}" -eq 1 ]]; then
    # Profile ALL ranks with unique output per global rank
    srun -N 2 --gpus-per-node=4 --ntasks-per-node=4 --export=ALL \
      bash -lc '
        out="ncu_reports/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_rank${SLURM_PROCID}"
        echo "Profiling rank $SLURM_PROCID on node $SLURM_NODEID -> $out.ncu-rep"
        touch "${out}.touch" 2>/dev/null || { echo "[Rank ${SLURM_PROCID}] Cannot write to $(pwd)/ncu_reports"; exit 1; }
        ncu $NCU_ARGS -o "$out" ./build/bench_hypre_gpu 2>"${out}.ncu.log"
      '
  else
    # Profile only local rank 0 per node
    srun -N 2 --gpus-per-node=4 --ntasks-per-node=4 --export=ALL \
      bash -lc '
        if [[ "$SLURM_LOCALID" -eq 0 ]]; then
          out="ncu_reports/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_rank${SLURM_PROCID}"
          echo "Profiling rank $SLURM_PROCID on node $SLURM_NODEID -> $out.ncu-rep"
          touch "${out}.touch" 2>/dev/null || { echo "[Rank ${SLURM_PROCID}] Cannot write to $(pwd)/ncu_reports"; exit 1; }
          ncu $NCU_ARGS -o "$out" ./build/bench_hypre_gpu 2>"${out}.ncu.log"
        else
          ./build/bench_hypre_gpu
        fi'
  fi
else
  srun -N 2 --gpus-per-node=4 --ntasks-per-node=4 \
    ./build/bench_hypre_gpu
fi
