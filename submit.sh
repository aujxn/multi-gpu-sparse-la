#!/bin/bash

BUILD_JOB_ID=$(sbatch --parsable build/build.slurm)

if [ -n "$BUILD_JOB_ID" ]; then
  echo "Build job $BUILD_JOB_ID submitted. Submitting run job with dependency..."
  sbatch --dependency=afterok:"$BUILD_JOB_ID" examples/simple.slurm
else
  echo "Failed to submit build job."
  exit 1
fi
