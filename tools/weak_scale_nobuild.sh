#!/usr/bin/env bash
set -euo pipefail

# Weak scaling study driver (no build). Submits 4 runs: 2, 4, 8, 16 GPUs.
# Usage examples:
#   RUN_TARGET=bench_poisson tools/weak_scale_nobuild.sh
#   RUN_TARGET=bench_hypre_gpu tools/weak_scale_nobuild.sh

RUN_TARGET=${RUN_TARGET:-bench_poisson}  # bench_poisson | bench_hypre_gpu

submit_run() {
  local nodes=$1 gpn=$2 nx=$3 cpt=$4
  local gpus=$((nodes*gpn))
  local name="ws_${RUN_TARGET}_G${gpus}"

  # sbatch options as arrays to preserve word splitting
  local -a export_opts=(
    "--export=ALL,DEBUG_MEM=1,GPU_ITERS=300,RUN_TARGET=${RUN_TARGET},RUN_NAME=${name},MESH_NX=${nx},SERIAL_REFS=2,PAR_REFS=3,NODES=${nodes},GPUS_PER_NODE=${gpn},NTASKS_PER_NODE=${gpn},CPUS_PER_TASK=${cpt}"
  )
  local -a res_opts=(
    "--job-name=${name}"
    "--nodes=${nodes}"
    "--gres=gpu:l40s:${gpn}"
    "--ntasks-per-node=${gpn}"
    "--cpus-per-task=${cpt}"
    "--exclusive"
    "--partition=short"
  )

  echo "Submitting ${name}: nodes=${nodes} gpn=${gpn} nx=${nx} cpt=${cpt}"
  sbatch "${export_opts[@]}" "${res_opts[@]}" benchmarks/bench_generic.slurm
}

# Map GPUs -> (nodes, gpn, nx, cpt).
# Baseline 4 GPUs uses nx=5. Scale nx ~ cube_root(gpus/4).
submit_run 1 2 9 32   # 1 Node, 2 GPUs, nx=9, cpus-per-task=32
submit_run 1 4 11 16  # 4 GPUs
submit_run 2 4 14 16  # 8 GPUs
submit_run 3 4 16 16  # 12 GPUs
submit_run 4 4 17 16  # 16 GPUs
submit_run 5 4 19 16  # 20 GPUs
submit_run 6 4 20 16  # 24 GPUs

echo "Submitted weak scaling runs (no build)"
