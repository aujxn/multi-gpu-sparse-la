#!/usr/bin/env bash

### usage:
##    # cpu only
##    `sbatch build_mfem.sh`
##    # a30 build
##    `sbatch --gres=gpu:a30:4 build_mfem.sh`
##    # for the l40s (different cuda arch a30 is sm_80 vs l40s is sm_89, detected automatically)
##    `sbatch --gres=gpu:l40s:4 build_mfem.sh`
###

#SBATCH --job-name=mfem_builder
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --partition=short
#SBATCH --output=logs/mfem_build.%j.out
#SBATCH --error=logs/mfem_build.%j.err
#SBATCH --time=01:00:00

# Forward selected environment variables to the build script so variant
# configuration is controlled by sbatch --export or inherited env.
# Examples:
#   sbatch --export=ALL,WITH_CUDA=1,WITH_MPI=1,GPU_AWARE=0,PRECISION=double,BUILD_TYPE=RelWithDebInfo --gres=gpu:l40s:4 build_mfem.slurm

# Ensure logs directory exists and purge any old build logs
mkdir -p logs
echo "[build_mfem.slurm] Purging old build logs in ./logs (keeping current job)"
# Keep current job's log files; remove others and legacy names.
# shellcheck disable=SC2034
# Current job's output/err paths (used to protect from deletion)
CUR_OUT="logs/mfem_build.${SLURM_JOB_ID}.out"
# shellcheck disable=SC2034
CUR_ERR="logs/mfem_build.${SLURM_JOB_ID}.err"
# Remove legacy names that won't be used by this job
rm -f logs/mfem_build.out logs/mfem_build.err 2>/dev/null || true
# Remove any prior job logs except the current job's files
find logs -maxdepth 1 -type f -name 'mfem_build.*.out' ! -name "mfem_build.${SLURM_JOB_ID}.out" -delete 2>/dev/null || true
find logs -maxdepth 1 -type f -name 'mfem_build.*.err' ! -name "mfem_build.${SLURM_JOB_ID}.err" -delete 2>/dev/null || true

echo "[build_mfem.slurm] Starting variant build at $(date)"
./build_mfem_script.sh
status=$?
echo "[build_mfem.slurm] Build script exited with code $status at $(date)"
exit $status
