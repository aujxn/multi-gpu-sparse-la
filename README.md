# Multi Node / Multi GPU Block Sparse Linear Algebra

## Overview

This repo serves as a testing ground to build a simple C API and ABI for distributed block sparse linear algebra BLAS types 2 and 3 operations (e.g. `spMM`/`spGEMM`). The core format is now Generic Blocked-ELL (BELL) with cuSPARSE SpMM.

## Usage

### Setup

`sbatch` scripts are for the `orca` accelerator. For now, you have to download your own static library files for `nccl`:
```
> ls nccl*
nccl_2.28.3-1+cuda12.9_x86_64.txz
> tar -xvf nccl*.txz
...
> export NCCL_HOME=$(pwd)/nccl_2.28.3-1+cuda12.9_x86_64
```

### Build

```bash
# Build via SLURM
sbatch build/build.slurm

# Or use the build script directly (loads modules automatically)
./build/build_script.sh
```

### Run

If you're on `orca`:

```bash
# Complete workflow: build then run simple example
./submit.sh

# Run individual examples (after build complete)
sbatch examples/simple.slurm
sbatch examples/dg.slurm
```

The simple example runs a toy `spMM` (n=1) with 2 nodes and 8 GPUs using BELL.

## License

MIT.

See `LICENSE.md`.
