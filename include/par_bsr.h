#pragma once
#include <stddef.h>
#include <stdint.h>
#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef long long gblk; // global block-column id

// Forward declares to avoid pulling CUDA/NCCL headers here
typedef struct CUstream_st* cudaStream_t;
typedef struct cusparseContext* cusparseHandle_t;
typedef struct ncclComm* ncclComm_t;

// Hypre-like distributed BSR container (device-resident matrices & vectors)
typedef struct {
  // Identity
  int pe;               // this rank
  int npes;             // world size

  // Block geometry (block-level)
  int mb_local;         // local block rows
  int nb_local;         // local block cols (square in demo)
  int nb_ghost;         // ghost block cols
  int bdim;             // block size (BSR_BDIM)

  // A_ii (local) BSR on device
  int    Aii_nnzb;
  int*   d_Aii_rowptr;  // len = mb_local+1
  int*   d_Aii_colind;  // len = Aii_nnzb
  float* d_Aii_val;     // len = Aii_nnzb*bdim*bdim
  void*  descrAii;      // opaque cusparseMatDescr_t

  // A_ij (off-diag, ghost) BSR on device
  int    Aij_nnzb;
  int*   d_Aij_rowptr;
  int*   d_Aij_colind;
  float* d_Aij_val;
  void*  descrAij;      // opaque cusparseMatDescr_t

  // Ghost mapping (host): global block-col id per ghost id [0..nb_ghost-1]
  int   nb_ghost_host;
  gblk* h_col_map_offd;

  // Global block ownership map (host): contiguous ranges in block units
  // length = npes+1, shared pointer ok
  gblk* h_col_starts;

  // Comm plan (host)
  int   n_neighbors;
  int*  neighbors;           // peer ranks
  int*  recv_counts;         // blocks per neighbor we receive
  int*  recv_offsets;        // displs into flattened recv arrays
  int*  send_counts;         // blocks per neighbor we send
  int*  send_offsets;

  // Flattened lists
  int*  h_recv_ghost_ids;        // len = sum(recv_counts)
  int*  h_send_local_block_ids;  // len = sum(send_counts)

  // Device mirrors and scratch
  int*   d_recv_ghost_ids;
  int*   d_send_local_block_ids;
  float* d_sendbuf;          // floats, len = sum(send_counts)*bdim
  float* d_recvbuf;          // floats, len = sum(recv_counts)*bdim

  // Distributed vectors
  float* d_x_local;          // len = mb_local*bdim
  float* d_x_ghost;          // len = nb_ghost*bdim
  float* d_y;                // len = mb_local*bdim

  // Handles
  cudaStream_t    stream_comm;
  cudaStream_t    stream_comp;
  cusparseHandle_t sp_handle;
  ncclComm_t      nccl;
} ParBSR;

/** Initialize container and CUDA/cuSPARSE streams/handles (no matrix yet). */
int parbsr_init(ParBSR* P,
                int pe, int npes,
                int mb_local, int bdim,
                gblk* col_starts,
                ncclComm_t nccl_comm);

/** Free all owned resources. */
void parbsr_free(ParBSR* P);

/** Build a tiny demo matrix:
 *  A_ii = 2*I blocks; A_ij = one 1-block row in the last block-row
 *  that references the FIRST block-col of the RIGHT neighbor (if any).
 *  Initializes x_local = 100*pe + [0..), zeros x_ghost,y.
 */
int parbsr_build_toy(ParBSR* P);

/** Build a ring comm plan:
 *  receive from right neighbor (if any) the ghost block(s) you need,
 *  send to left neighbor the corresponding local block(s).
 *  (Matches the toy A_ij.)
 */
int parbsr_build_comm_plan_ring(ParBSR* P);


/** Build a general comm-plan from P->h_col_map_offd using MPI:
 *  - Computes neighbors (both send & recv)
 *  - Sends requested global block-ids to owners (Alltoallv)
 *  - Owners map to local block ids → send lists
 *  - Receivers keep ghost-id order → recv lists
 */
int parbsr_build_comm_plan_from_colmap(ParBSR* P, MPI_Comm comm);

/** Pull halo for x into x_ghost (pack → NCCL send/recv → unpack). */
int parbsr_halo_x(ParBSR* P);

/** y = A_ii*x_local + A_ij*x_ghost */
int parbsr_spmv(ParBSR* P);

/** Debug print (host pull) */
void parbsr_print_vec(const ParBSR* P, const char* name, const float* dptr, int n);

#ifdef __cplusplus
} // extern "C"
#endif
