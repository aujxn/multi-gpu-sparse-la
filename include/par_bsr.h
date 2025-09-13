// Compatibility shim for legacy BSR naming (used by MFEM example).
// Maps ParBSR/parbsr_* to ParBELL/parbell_* symbols.

#pragma once
#include "par_bell.h"

typedef ParBELL ParBSR;

#define parbsr_init  parbell_init
#define parbsr_free  parbell_free
#define parbsr_build_comm_plan_from_colmap parbell_build_comm_plan_from_colmap
#define parbsr_halo_x parbell_halo_x
#define parbsr_spmv  parbell_spmv
#define parbsr_print_vec parbell_print_vec

// Note: MFEM/Hypre helper functions remain in src/par_bsr_mfem.cpp and are
// intentionally not renamed here.
