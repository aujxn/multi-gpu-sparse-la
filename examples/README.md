# Examples

## simple.cpp

Basic multi-GPU sparse matrix-vector multiply (via SpMM with n=1) using toy matrices. Demonstrates the core ParBELL functionality with a simple ring communication pattern and BELL storage.

**Build:** `make simple_example`  
**Run:** `sbatch run.sh` (configured for simple_example)

## dg_diffusion.cpp

MFEM/Hypre integration example that:
- Creates a DG discretization of the Poisson problem using MFEM
- Extracts the parallel matrix from Hypre ParCSR format 
- Converts to our ParBELL format (CSR→BELL for now with block size 1)
- Performs SpMV with our library and verifies against Hypre reference

**Dependencies:** Requires MFEM and Hypre built with MPI support  
**Build:** `make dg_diffusion` (if MFEM/Hypre found)  
**Run:** `srun -N 2 --ntasks-per-node=4 --mpi=pmix ./build/dg_diffusion`

**Note:** Currently assumes block size = 1. Future work will extend to true block sparse formats.
