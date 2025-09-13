#pragma once
#include <stddef.h>
#include <stdint.h>
#include <mpi.h>

#include "comm_pkg.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef long long gblk; // global block-column id

// Forward declares to avoid pulling CUDA/NCCL headers here
typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st*  cudaEvent_t;
typedef struct cusparseContext* cusparseHandle_t;
typedef struct ncclComm* ncclComm_t;

// Distributed Blocked-ELL container (device-resident matrices & vectors)
typedef struct {
  // Identity
  int pe;               // this rank
  int npes;             // world size

  // Block geometry (block-level)
  int mb_local;         // local block rows
  int nb_local;         // local block cols (square in demo)
  int nb_ghost;         // ghost block cols
  int bdim;             // block size

  // A_ii (local) Blocked-ELL on device
  int    Aii_ell_cols;      // ELL slots per block-row
  int*   d_Aii_colind;      // len = mb_local * Aii_ell_cols (int32)
  float* d_Aii_val;         // len = mb_local * Aii_ell_cols * bdim * bdim
  void*  matAii;            // opaque cusparseSpMatDescr_t

  // A_ij (off-diag, ghost) Blocked-ELL on device
  int    Aij_ell_cols;      // ELL slots per block-row (0 or >0)
  int*   d_Aij_colind;      // len = mb_local * Aij_ell_cols
  float* d_Aij_val;         // len = mb_local * Aij_ell_cols * bdim * bdim
  void*  matAij;            // opaque cusparseSpMatDescr_t

  // Ghost mapping (host): global block-col id per ghost id [0..nb_ghost-1]
  int   nb_ghost_host;
  gblk* h_col_map_offd;

  // Global block ownership map (host): contiguous ranges in block units
  // length = npes+1, shared pointer ok
  gblk* h_col_starts;

  // Communication plan (reusable indices on host+device)
  CommPlan plan;

  // Scratch buffers for halo exchange (data only)
  float* d_sendbuf;          // floats, len = sum(send_counts)*bdim
  float* d_recvbuf;          // floats, len = sum(recv_counts)*bdim

  // Distributed vectors
  float* d_x_local;          // len = mb_local*bdim
  float* d_x_ghost;          // len = nb_ghost*bdim
  float* d_y;                // len = mb_local*bdim

  // Dense matrix descriptors for SpMM (ncols = 1)
  void*  dnX_local;          // opaque cusparseDnMatDescr_t (Kx1)
  void*  dnX_ghost;          // opaque cusparseDnMatDescr_t (Kx1)
  void*  dnY;                // opaque cusparseDnMatDescr_t (Mx1)
  // Split-parallel outputs (optional)
  float* d_y_aii;            // len = mb_local*bdim (temp)
  float* d_y_aij;            // len = mb_local*bdim (temp)
  void*  dnY_aii;            // opaque cusparseDnMatDescr_t (Mx1) for d_y_aii
  void*  dnY_aij;            // opaque cusparseDnMatDescr_t (Mx1) for d_y_aij
  // Separate workspaces per sparse matrix
  void*  d_workspace_aii;
  void*  d_workspace_aij;
  size_t workspace_size_aii;
  size_t workspace_size_aij;

  // Handles
  cudaStream_t     stream_comm;
  cudaStream_t     stream_comp;
  cudaStream_t     stream_comp2;   // second compute stream (Aij)
  cusparseHandle_t sp_handle;
  cusparseHandle_t sp_handle_aij;  // second handle bound to stream_comp2
  ncclComm_t      nccl;
  cudaEvent_t     evt_halo_done;   // halo completion event (for overlap)
} ParBELL;

/** Initialize container and CUDA/cuSPARSE streams/handles (no matrix yet). */
int parbell_init(ParBELL* P,
                int pe, int npes,
                int mb_local, int bdim,
                gblk* col_starts,
                ncclComm_t nccl_comm);

/** Free all owned resources. */
void parbell_free(ParBELL* P);



/** Build a general comm-plan from P->h_col_map_offd using MPI. */
int parbell_build_comm_plan_from_colmap(ParBELL* P, MPI_Comm comm);

/** Pull halo for x into x_ghost (pack → NCCL send/recv → unpack). */
int parbell_halo_x(ParBELL* P);

/** y = A_ii*x_local + A_ij*x_ghost (via SpMM with n=1) */
int parbell_spmv(ParBELL* P);

/** Upload Blocked-ELL matrix data to device */
int upload_blocked_ell(int mb, int ell_cols, int bdim,
                       const int *h_colind, const float *h_val,
                       int **d_colind, float **d_val);

/** Create modern cuSPARSE API descriptors for matrices and vectors. */
int create_cusparse_descriptors(ParBELL* P);

/** Debug print (host pull) */
void parbell_print_vec(const ParBELL* P, const char* name, const float* dptr, int n);

#ifdef __cplusplus
} // extern "C"
#endif
