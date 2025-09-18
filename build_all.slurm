#!/usr/bin/env bash

#SBATCH --job-name=build_all
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --partition=short
#SBATCH --output=logs/build_all.%j.out
#SBATCH --error=logs/build_all.%j.err
#SBATCH --time=01:00:00

set -euo pipefail

mkdir -p logs
echo "[build_all.slurm] Purging old build_all logs (keeping current job)"
# shellcheck disable=SC2034
CUR_OUT="logs/build_all.${SLURM_JOB_ID}.out"
# shellcheck disable=SC2034
CUR_ERR="logs/build_all.${SLURM_JOB_ID}.err"
find logs -maxdepth 1 -type f -name 'build_all.*.out' ! -name "build_all.${SLURM_JOB_ID}.out" -delete 2>/dev/null || true
find logs -maxdepth 1 -type f -name 'build_all.*.err' ! -name "build_all.${SLURM_JOB_ID}.err" -delete 2>/dev/null || true

echo "[build_all.slurm] Starting at $(date)"
bash build_all_script.sh
status=$?
echo "[build_all.slurm] Script exited with code $status at $(date)"
exit $status
