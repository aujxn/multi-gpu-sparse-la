#!/usr/bin/env bash
#SBATCH --partition=short
#SBATCH --exclusive
#SBATCH --mem=500G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:30

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs ncu_reports

module load cuda || true

# Networking for NCCL
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ens2,ib0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# Conservative defaults for memory usage unless overridden
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OMP_STACKSIZE=${OMP_STACKSIZE:-2M}
export DEBUG_MEM=${DEBUG_MEM:-0}

# OpenMP threading: align with Slurm cpus-per-task if provided
THREADS=${SLURM_CPUS_PER_TASK:-${CPUS_PER_TASK:-${OMP_NUM_THREADS:-1}}}
export OMP_NUM_THREADS=${THREADS}
export OMP_PROC_BIND=${OMP_PROC_BIND:-spread}
export OMP_PLACES=${OMP_PLACES:-cores}
export OMP_DYNAMIC=${OMP_DYNAMIC:-false}

# Mesh parameters
export MESH_NX=${MESH_NX:-5}
export MESH_NY=${MESH_NY:-$MESH_NX}
export MESH_NZ=${MESH_NZ:-$MESH_NX}
export SERIAL_REFS=${SERIAL_REFS:-2}
export PAR_REFS=${PAR_REFS:-3}

# Run target
TARGET=${RUN_TARGET:-bench_poisson}
BIN=./build/${TARGET}
if [[ ! -x "$BIN" ]]; then echo "Binary not found: $BIN" >&2; fi
export BIN

# Resources (must be provided via sbatch CLI)
NODES=${NODES:-${SLURM_JOB_NUM_NODES:-1}}
GPUS_PER_NODE=${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}
NTASKS_PER_NODE=${NTASKS_PER_NODE:-$GPUS_PER_NODE}
CPUS_PER_TASK=${CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK:-1}}

# Optional Nsight Compute profiling
if [[ "${PROFILE_WITH_NCU:-0}" -eq 1 ]]; then
  which ncu || true
  ncu --version || true
  REQ_SET="${NCU_SET:-speedOfLight}"
  NCU_VERBOSE=${NCU_VERBOSE:-2}
  export NCU_ARGS="--set ${REQ_SET} --replay-mode application --force-overwrite true --kernel-name-base demangled --profile-from-start on --verbose ${NCU_VERBOSE} ${NCU_ARGS_EXTRA:-}"
  srun -N "$NODES" --gpus-per-node="$GPUS_PER_NODE" --ntasks-per-node="$NTASKS_PER_NODE" --mpi=pmix --mem=${SRUN_MEM:-0} --export=ALL \
    bash -lc '
      out="ncu_reports/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_rank${SLURM_PROCID}"
      echo "Profiling rank $SLURM_PROCID -> $out.ncu-rep"
      ncu $NCU_ARGS -o "$out" "$BIN"
    '
else
  srun -N "$NODES" --gpus-per-node="$GPUS_PER_NODE" --ntasks-per-node="$NTASKS_PER_NODE" --mpi=pmix --mem=${SRUN_MEM:-0} --export=ALL \
    "$BIN"
fi
