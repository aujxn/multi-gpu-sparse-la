#!/usr/bin/env bash
set -euo pipefail

# Strong scaling study driver (no build).
# Uses the largest weak-scaling problem: nx=8, SERIAL_REFS=2, PAR_REFS=3
# Submits 2, 4, 8, 16 GPU runs with a fixed mesh size.

RUN_TARGET=${RUN_TARGET:-bench_poisson}  # bench_poisson | bench_hypre_gpu

submit_run() {
  local nodes=$1 gpn=$2 nx=$3 cpt=$4
  local gpus=$((nodes*gpn))
  local name="ss_${RUN_TARGET}_G${gpus}"

  # Environment for fixed problem size; also skip CPU reference to save time
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

# Strong scaling with fixed problem size
submit_run 1 2 11 32    # 2 GPUs
submit_run 1 4 11 16    # 4 GPUs
submit_run 2 4 11 16    # 8 GPUs
submit_run 3 4 11 16    # 12 GPUs
submit_run 4 4 11 16    # 16 GPUs
submit_run 5 4 11 16    # 20 GPUs
submit_run 6 4 11 16    # 24 GPUs

echo "Submitted strong scaling runs (no build)"
