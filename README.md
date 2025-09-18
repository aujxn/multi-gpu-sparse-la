# Multi Node / Multi GPU Block Sparse Linear Algebra

## Overview

This repo serves as a testing ground to build a simple C API and ABI for distributed block sparse linear algebra BLAS types 2 and 3 operations (e.g. `spMM`/`spGEMM`). The core format is now Generic Blocked-ELL (BELL) with cuSPARSE SpMM.

## Usage

### Dependency Builds (MFEM + Hypre + METIS)

Our SLURM builder creates per-variant installs under `/scratch/ajn6-amg/multi_gpu/deps/<variant>/...`.

Variants are controlled with environment variables; typical settings for L40S:

```bash
# Single-precision, CUDA enabled, MPI enabled, non-GPU-aware MPI, then configure+build project
sbatch --export=ALL,WITH_MPI=1,WITH_CUDA=1,GPU_AWARE=0,PRECISION=single,BUILD_TYPE=RelWithDebInfo \
       --gres=gpu:l40s:4 build_all.slurm
```

Notes:
- Sources are expected under `$HOME/opt/src/{hypre,mfem,metis-5.1.0}`.
- Build logs: `logs/mfem_build.$JOBID.{out,err}`.
- After successful build, the script prints the chosen `HYPRE_DIR` and `MFEM_DIR` (or MFEM source) to use below.

### Project Configure (CMake)

Use the helper to configure CMake. Set `NCCL_HOME` if not auto-discoverable.

```bash
# Optional: manual configure/build if you didn't use build_all.slurm
# export NCCL_HOME=$HOME/opt/src/nccl_2.28.3-1+cuda12.9_x86_64
# ./configure.sh --hypre-dir /scratch/.../deps/<variant>/hypre --mfem-dir $HOME/opt/src/mfem --type RelWithDebInfo --cuda-arch 89
# cmake --build build -j
```

### Run

If you're on `orca`:

```bash
# Complete workflow: build then run simple example (afterok dependency)
./submit.sh

# Run individual jobs (after project build)
sbatch examples/simple.slurm
sbatch examples/dg.slurm
sbatch benchmarks/poisson_bench.slurm
sbatch benchmarks/poisson_bench_hypre_gpu.slurm
```

The simple example runs a toy `spMM` (n=1) with 2 nodes and 8 GPUs using BELL.

## License

MIT.

See `LICENSE.md`.
