#!/usr/bin/env bash
set -euo pipefail

# Weak scaling study driver.
# Submits a build job once, then four run jobs (2,4,8,16 GPUs) with afterok dependency.

RUN_TARGET=${RUN_TARGET:-bench_poisson}  # bench_poisson | bench_hypre_gpu
BUILD_ARGS=${BUILD_SBATCH_ARGS:-}

submit_build() {
  echo "Submitting build_all.slurm with: ${BUILD_ARGS:-<none>}"
  if [[ -n "$BUILD_ARGS" ]]; then
    # shellcheck disable=SC2086
    sbatch $BUILD_ARGS --parsable build_all.slurm
  else
    sbatch --parsable build_all.slurm
  fi
}

submit_run() {
  local nodes=$1 gpn=$2 nx=$3 cpt=$4
  local gpus=$((nodes*gpn))
  local name="ws_${RUN_TARGET}_G${gpus}"
  local -a export_opts=(
    "--export=ALL,RUN_TARGET=${RUN_TARGET},RUN_NAME=${name},NODES=${nodes},GPUS_PER_NODE=${gpn},NTASKS_PER_NODE=${gpn},CPUS_PER_TASK=${cpt},MESH_NX=${nx},SERIAL_REFS=2,PAR_REFS=3"
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
  echo "Submitting run: ${name} -> nodes=${nodes} gpn=${gpn} nx=${nx} cpt=${cpt}"
  sbatch --dependency=afterok:"${BUILD_JOB_ID}" "${export_opts[@]}" "${res_opts[@]}" benchmarks/bench_generic.slurm
}

# Submit build
BUILD_JOB_ID=$(submit_build)
if [[ -z "$BUILD_JOB_ID" ]]; then echo "Build submission failed" >&2; exit 1; fi
echo "Build job id: $BUILD_JOB_ID"

# Map GPUs -> (nodes, gpn, nx, cpt)
# Baseline 4 GPUs uses nx=5. Scale nx ~ cube_root(gpus/4).
submit_run 1 2 4 32   # 2 GPUs: nx=4, cpus-per-task=32
submit_run 1 4 5 16   # 4 GPUs: nx=5
submit_run 2 4 6 16   # 8 GPUs: nx=6
submit_run 4 4 8 16   # 16 GPUs: nx=8

echo "Submitted weak scaling runs with dependency on build job ${BUILD_JOB_ID}" 
