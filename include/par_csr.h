#pragma once
#include <stddef.h>
#include <stdint.h>
#include <mpi.h>

#include "comm_pkg.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declares to avoid pulling CUDA/NCCL headers here
typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st*  cudaEvent_t;
typedef struct cusparseContext* cusparseHandle_t;
typedef struct ncclComm* ncclComm_t;

// Distributed CSR container (device-resident matrices & vectors)
typedef struct {
  // Identity
  int pe;
  int npes;

  // Geometry (scalar)
  int m_local;  // local rows
  int n_local;  // local cols owned by this rank
  int n_ghost;  // remote cols referenced

  // CSR A_ii (local columns)
  int*   d_Aii_rowptr;   // len = m_local+1
  int*   d_Aii_colind;   // len = nnzAii (local col index [0..n_local-1])
  float* d_Aii_val;      // len = nnzAii
  void*  matAii;         // cusparseSpMatDescr_t

  // CSR A_ij (ghost columns)
  int*   d_Aij_rowptr;   // len = m_local+1
  int*   d_Aij_colind;   // len = nnzAij (ghost col index [0..n_ghost-1])
  float* d_Aij_val;      // len = nnzAij
  void*  matAij;         // cusparseSpMatDescr_t

  // Ghost mapping (host)
  int   n_ghost_host;
  gblk* h_col_map_offd;  // length n_ghost_host, maps ghost id -> global col id
  gblk* h_col_starts;    // length npes+1, ownership map

  // Communication plan
  CommPlan plan;

  // Device vectors
  float* d_x_local;  // len = n_local
  float* d_x_ghost;  // len = n_ghost
  float* d_y;        // len = m_local

  // Scratch for halo exchange
  float* d_sendbuf;  // len = total_send
  float* d_recvbuf;  // len = total_recv

  // cuSPARSE descriptors for SpMV
  void*  dnX_local;  // cusparseDnVecDescr_t
  void*  dnX_ghost;  // cusparseDnVecDescr_t
  void*  dnY;        // cusparseDnVecDescr_t (standard path)
  // Optional split-parallel outputs
  float* d_y_aii;    // len = m_local (temp)
  float* d_y_aij;    // len = m_local (temp)
  void*  dnY_aii;    // cusparseDnVecDescr_t (points to d_y_aii)
  void*  dnY_aij;    // cusparseDnVecDescr_t (points to d_y_aij)
  // Separate workspaces per sparse matrix (default)
  void*  d_workspace_aii;
  void*  d_workspace_aij;
  size_t workspace_size_aii;
  size_t workspace_size_aij;

  // Handles
  cudaStream_t     stream_comm;
  cudaStream_t     stream_comp;
  cudaStream_t     stream_comp2;   // optional second compute stream
  cusparseHandle_t sp_handle;
  cusparseHandle_t sp_handle_aij;  // optional second handle (for split-parallel)
  ncclComm_t       nccl;
  cudaEvent_t      evt_halo_done;  // signals halo completion on stream_comm
} ParCSR;

int parcsr_init(ParCSR* P,
                int pe, int npes,
                int m_local, int n_local,
                gblk* col_starts,
                ncclComm_t nccl_comm);

void parcsr_free(ParCSR* P);

int parcsr_build_comm_plan_from_colmap(ParCSR* P, MPI_Comm comm);

int parcsr_halo_x(ParCSR* P);

int parcsr_spmv(ParCSR* P);

// Upload CSR arrays to device
int upload_csr(int m, int nnz,
               const int* h_rowptr, const int* h_colind, const float* h_val,
               int** d_rowptr, int** d_colind, float** d_val);

// Create cuSPARSE descriptors (CSR mats + DnVecs) and workspaces
int parcsr_create_cusparse_descriptors(ParCSR* P, int nnzAii, int nnzAij);

// Debug print (host pull)
void parcsr_print_vec(const ParCSR* P, const char* name, const float* dptr, int n);

#ifdef __cplusplus
} // extern "C"
#endif
