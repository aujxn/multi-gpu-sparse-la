#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./submit.sh                              # build, then run examples/simple.slurm afterok
#   RUN_BATCH=benchmarks/poisson_bench.slurm ./submit.sh

RUN_BATCH=${RUN_BATCH:-examples/simple.slurm}
BUILD_SBATCH_ARGS=${BUILD_SBATCH_ARGS:-}

echo "Submitting build job with variant env (subset shown):"
echo "  WITH_MPI=${WITH_MPI:-} WITH_CUDA=${WITH_CUDA:-} GPU_AWARE=${GPU_AWARE:-} PRECISION=${PRECISION:-} BUILD_TYPE=${BUILD_TYPE:-}"
if [[ -n "$BUILD_SBATCH_ARGS" ]]; then echo "  sbatch extra args: $BUILD_SBATCH_ARGS"; fi

if [[ -n "$BUILD_SBATCH_ARGS" ]]; then
  # shellcheck disable=SC2086
  BUILD_JOB_ID=$(sbatch $BUILD_SBATCH_ARGS --parsable build_all.slurm)
else
  BUILD_JOB_ID=$(sbatch --parsable build_all.slurm)
fi
if [[ -z "$BUILD_JOB_ID" ]]; then
  echo "Failed to submit build job." >&2
  exit 1
fi
echo "Build job $BUILD_JOB_ID submitted."

echo "Submitting run job $RUN_BATCH with dependency afterok:$BUILD_JOB_ID …"
sbatch --dependency=afterok:"$BUILD_JOB_ID" "$RUN_BATCH"
