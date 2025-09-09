# Multi Node / Multi GPU Block Sparse Linear Algebra

## Overview

This repo serves as a testing ground to build a simple C API and ABI for distributed block sparse linear algebra BLAS types 2 and 3 operations (e.g. `spMM`/`spGEMM`).

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

### Run

If you're on `orca` then `./submit.sh` runs a toy example `spMV` with 2 nodes and 8 GPUs.

## License

MIT.

See `LICENSE.md`.
