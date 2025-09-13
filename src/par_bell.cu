#include "par_bell.h"
#include "utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

 

// ---------------- kernels: pack/unpack blocks ----------------
__global__ void pack_blocks_f32(const float *__restrict__ x_local,
                                float *__restrict__ sendbuf,
                                const int *__restrict__ block_ids, int nblocks,
                                int bdim) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int N = nblocks * bdim;
  if (t >= N)
    return;
  int b = t / bdim;
  int r = t % bdim;
  int src = block_ids[b] * bdim + r;
  sendbuf[t] = x_local[src];
}

__global__ void unpack_blocks_f32(const float *__restrict__ recvbuf,
                                  float *__restrict__ x_ghost,
                                  const int *__restrict__ ghost_ids,
                                  int nblocks, int bdim) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int N = nblocks * bdim;
  if (t >= N)
    return;
  int b = t / bdim;
  int r = t % bdim;
  int dst = ghost_ids[b] * bdim + r;
  x_ghost[dst] = recvbuf[t];
}

// ---------------- API impl ----------------

int parbell_init(ParBELL *P, int pe, int npes, int mb_local, int bdim,
                gblk *col_starts, ncclComm_t nccl_comm) {
  if (!P)
    return -1;
  memset(P, 0, sizeof(*P));
  P->pe = pe;
  P->npes = npes;
  P->mb_local = mb_local;
  P->nb_local = mb_local; // square in demo
  P->bdim = bdim;
  P->h_col_starts = col_starts;
  P->nccl = nccl_comm;

  // streams + cuSPARSE
  CHECK_CUDA(cudaStreamCreate(&P->stream_comm));
  CHECK_CUDA(cudaStreamCreate(&P->stream_comp));
  CHECK_CUDA(cudaStreamCreate(&P->stream_comp2));
  CHECK_CUSPARSE(cusparseCreate(&P->sp_handle));
  CHECK_CUSPARSE(cusparseSetStream(P->sp_handle, P->stream_comp));
  CHECK_CUSPARSE(cusparseCreate(&P->sp_handle_aij));
  CHECK_CUSPARSE(cusparseSetStream(P->sp_handle_aij, P->stream_comp2));
  CHECK_CUDA(cudaEventCreateWithFlags(&P->evt_halo_done, cudaEventDisableTiming));

  // Initialize modern API descriptors (will be created later when matrix data is available)
  P->matAii = nullptr;
  P->matAij = nullptr;
  P->dnX_local = nullptr;
  P->dnX_ghost = nullptr;
  P->dnY = nullptr;
  P->d_y_aii = nullptr; P->d_y_aij = nullptr;
  P->dnY_aii = nullptr; P->dnY_aij = nullptr;
  P->d_workspace_aii = nullptr; P->d_workspace_aij = nullptr;
  P->workspace_size_aii = 0; P->workspace_size_aij = 0;

  return 0;
}

int upload_blocked_ell(int mb, int ell_cols, int bdim,
                       const int *h_colind, const float *h_val,
                       int **d_colind, float **d_val) {
  int cols_elems = mb * ell_cols;
  int val_elems = cols_elems * bdim * bdim;
  CHECK_CUDA(cudaMalloc((void **)d_colind, (cols_elems > 0 ? cols_elems : 1) * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void **)d_val,    (val_elems  > 0 ? val_elems  : 1) * sizeof(float)));
  if (cols_elems > 0)
    CHECK_CUDA(cudaMemcpy(*d_colind, h_colind, cols_elems * sizeof(int), cudaMemcpyHostToDevice));
  if (val_elems > 0)
    CHECK_CUDA(cudaMemcpy(*d_val, h_val, val_elems * sizeof(float), cudaMemcpyHostToDevice));
  return 0;
}

// Helper function to create cuSPARSE Blocked-ELL + DnMat descriptors
int create_cusparse_descriptors(ParBELL *P) {
  // Create Blocked-ELL mats
  // rows/cols must be SCALAR dims (M, K)
  // arg order for cusparseCreateBlockedEll: (rows, cols, ellBlockSize, ellCols, ...)
  if (P->Aii_ell_cols > 0) {
    int64_t m = (int64_t)P->mb_local * P->bdim;
    int64_t k = (int64_t)P->nb_local * P->bdim;
    cusparseStatus_t s = cusparseCreateBlockedEll((cusparseSpMatDescr_t *)&P->matAii,
                                                  m, k,
                                                  (int64_t)P->bdim,                         // ellBlockSize
                                                  (int64_t)P->Aii_ell_cols * P->bdim,       // ellCols (scalar)
                                                  P->d_Aii_colind,
                                                  P->d_Aii_val,
                                                  CUSPARSE_INDEX_32I,
                                                  CUSPARSE_INDEX_BASE_ZERO,
                                                  CUDA_R_32F);
    if (s != CUSPARSE_STATUS_SUCCESS) {
      debug_logf(P->pe, "cusparseCreateBlockedEll(Aii) failed status=%d", (int)s);
      return -1;
    }
  }
  if (P->Aij_ell_cols > 0 && P->nb_ghost > 0) {
    int64_t m = (int64_t)P->mb_local * P->bdim;
    int64_t k = (int64_t)P->nb_ghost * P->bdim;
    cusparseStatus_t s = cusparseCreateBlockedEll((cusparseSpMatDescr_t *)&P->matAij,
                                                  m, k,
                                                  (int64_t)P->bdim,                         // ellBlockSize
                                                  (int64_t)P->Aij_ell_cols * P->bdim,       // ellCols (scalar)
                                                  P->d_Aij_colind,
                                                  P->d_Aij_val,
                                                  CUSPARSE_INDEX_32I,
                                                  CUSPARSE_INDEX_BASE_ZERO,
                                                  CUDA_R_32F);
    if (s != CUSPARSE_STATUS_SUCCESS) {
      debug_logf(P->pe, "cusparseCreateBlockedEll(Aij) failed status=%d", (int)s);
      return -1;
    }
  }

  // Dense matrices for SpMM (n=1)
  int64_t mY = (int64_t)P->mb_local * P->bdim;
  int64_t n = 1;
  int64_t k_local = (int64_t)P->nb_local * P->bdim;
  int64_t k_ghost = (int64_t)P->nb_ghost * P->bdim;
  // Create dense descriptors for SpMM (n=1), no verbose logging here
  {
    cusparseStatus_t s1 = cusparseCreateDnMat((cusparseDnMatDescr_t *)&P->dnX_local,
                                              k_local, n, k_local, (void *)P->d_x_local,
                                              CUDA_R_32F, CUSPARSE_ORDER_COL);
    if (s1 != CUSPARSE_STATUS_SUCCESS) { debug_logf(P->pe, "CreateDnMat X_local failed %d", (int)s1); return -1; }
    cusparseStatus_t s2 = cusparseCreateDnMat((cusparseDnMatDescr_t *)&P->dnY,
                                              mY, n, mY, (void *)P->d_y,
                                              CUDA_R_32F, CUSPARSE_ORDER_COL);
    if (s2 != CUSPARSE_STATUS_SUCCESS) { debug_logf(P->pe, "CreateDnMat Y failed %d", (int)s2); return -1; }
    cusparseStatus_t s3 = cusparseCreateDnMat((cusparseDnMatDescr_t *)&P->dnX_ghost,
                                              (k_ghost > 0 ? k_ghost : 1), n,
                                              (k_ghost > 0 ? k_ghost : 1), (void *)P->d_x_ghost,
                                              CUDA_R_32F, CUSPARSE_ORDER_COL);
    if (s3 != CUSPARSE_STATUS_SUCCESS) { debug_logf(P->pe, "CreateDnMat X_ghost failed %d", (int)s3); return -1; }
  }

  // Workspaces per matrix (Aii, Aij)
  if (P->matAii) {
    const float alpha = 1.0f, beta = 0.0f;
    size_t sz = 0;
    cusparseStatus_t sb = cusparseSpMM_bufferSize(P->sp_handle,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  &alpha, (cusparseSpMatDescr_t)P->matAii,
                                                  (cusparseDnMatDescr_t)P->dnX_local,
                                                  &beta, (cusparseDnMatDescr_t)P->dnY,
                                                  CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                                                  &sz);
    if (sb != CUSPARSE_STATUS_SUCCESS) { debug_logf(P->pe, "SpMM_bufferSize(Aii) failed %d", (int)sb); return -1; }
    P->workspace_size_aii = sz;
    if (sz > 0) CHECK_CUDA(cudaMalloc(&P->d_workspace_aii, sz));
  }
  if (P->matAij && P->nb_ghost > 0) {
    const float alpha = 1.0f, beta = 1.0f;
    size_t sz = 0;
    cusparseStatus_t sb = cusparseSpMM_bufferSize(P->sp_handle_aij,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  &alpha, (cusparseSpMatDescr_t)P->matAij,
                                                  (cusparseDnMatDescr_t)P->dnX_ghost,
                                                  &beta, (cusparseDnMatDescr_t)P->dnY,
                                                  CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                                                  &sz);
    if (sb != CUSPARSE_STATUS_SUCCESS) { debug_logf(P->pe, "SpMM_bufferSize(Aij) failed %d", (int)sb); return -1; }
    P->workspace_size_aij = sz;
    if (sz > 0) CHECK_CUDA(cudaMalloc(&P->d_workspace_aij, sz));
  }
  // Preprocess for repeated SpMM
  if (P->matAii) {
    const float alpha = 1.0f, beta = 0.0f;
    cusparseStatus_t sp = cusparseSpMM_preprocess(P->sp_handle,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  &alpha, (cusparseSpMatDescr_t)P->matAii,
                                                  (cusparseDnMatDescr_t)P->dnX_local,
                                                  &beta, (cusparseDnMatDescr_t)P->dnY,
                                                  CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                                                  P->d_workspace_aii);
    if (sp != CUSPARSE_STATUS_SUCCESS) { debug_logf(P->pe, "SpMM_preprocess(Aii) failed %d", (int)sp); return -1; }
  }
  if (P->matAij && P->nb_ghost > 0) {
    const float alpha = 1.0f, beta = 1.0f;
    cusparseStatus_t sp = cusparseSpMM_preprocess(P->sp_handle_aij,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  &alpha, (cusparseSpMatDescr_t)P->matAij,
                                                  (cusparseDnMatDescr_t)P->dnX_ghost,
                                                  &beta, (cusparseDnMatDescr_t)P->dnY,
                                                  CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                                                  P->d_workspace_aij);
    if (sp != CUSPARSE_STATUS_SUCCESS) { debug_logf(P->pe, "SpMM_preprocess(Aij) failed %d", (int)sp); return -1; }
  }
  return 0;
}

 

 

int parbell_build_comm_plan_from_colmap(ParBELL *P, MPI_Comm comm) {
  int rc = comm_plan_build_from_colmap(&P->plan, P->pe, P->npes,
                                       P->nb_ghost_host,
                                       P->h_col_map_offd,
                                       P->h_col_starts,
                                       comm);
  if (rc != 0) return rc;
  int total_recv_blocks = comm_plan_total_recv(&P->plan);
  int total_send_blocks = comm_plan_total_send(&P->plan);
  if (total_send_blocks > 0) CHECK_CUDA(cudaMalloc((void**)&P->d_sendbuf, (size_t)total_send_blocks * (size_t)P->bdim * sizeof(float)));
  if (total_recv_blocks > 0) CHECK_CUDA(cudaMalloc((void**)&P->d_recvbuf, (size_t)total_recv_blocks * (size_t)P->bdim * sizeof(float)));
  return 0;
}

int parbell_halo_x(ParBELL *P) {
  CHECK_CUDA(cudaMemsetAsync(P->d_x_ghost, 0,
                             (size_t)P->nb_ghost * P->bdim * sizeof(float),
                             P->stream_comm));

  // Pack local blocks → sendbuf
  int total_send = comm_plan_total_send(&P->plan);
  int total_recv = comm_plan_total_recv(&P->plan);

  int threads = 128;
  int s_off = 0;
  for (int a = 0; a < P->plan.n_neighbors; ++a) {
    int cnt = P->plan.send_counts[a];
    int n = cnt * P->bdim;
    if (n > 0) {
      pack_blocks_f32<<<(n + threads - 1) / threads, threads, 0,
                        P->stream_comm>>>(
          P->d_x_local, P->d_sendbuf + s_off * P->bdim,
          P->plan.d_send_ids + P->plan.send_offsets[a], cnt, P->bdim);
    }
    s_off += cnt;
  }

  // NCCL exchange (grouped per peer: recv then send)
  CHECK_NCCL(ncclGroupStart());
  int r_off = 0; s_off = 0;
  for (int a = 0; a < P->plan.n_neighbors; ++a) {
    int peer = P->plan.neighbors[a];
    int cnt_r = P->plan.recv_counts[a] * P->bdim;
    int cnt_s = P->plan.send_counts[a] * P->bdim;
    if (cnt_r > 0)
      CHECK_NCCL(ncclRecv(P->d_recvbuf + r_off, cnt_r, ncclFloat, peer, P->nccl,
                          P->stream_comm));
    if (cnt_s > 0)
      CHECK_NCCL(ncclSend(P->d_sendbuf + s_off, cnt_s, ncclFloat, peer, P->nccl,
                          P->stream_comm));
    r_off += cnt_r;
    s_off += cnt_s;
  }
  CHECK_NCCL(ncclGroupEnd());

  // Unpack recvbuf → x_ghost
  r_off = 0;
  for (int a = 0; a < P->plan.n_neighbors; ++a) {
    int cnt = P->plan.recv_counts[a];
    int n = cnt * P->bdim;
    if (n > 0) {
      unpack_blocks_f32<<<(n + threads - 1) / threads, threads, 0,
                          P->stream_comm>>>(
          P->d_recvbuf + r_off, P->d_x_ghost,
          P->plan.d_recv_ids + P->plan.recv_offsets[a], cnt, P->bdim);
    }
    r_off += n;
  }

  // Record halo completion to enable compute overlap
  CHECK_CUDA(cudaEventRecord(P->evt_halo_done, P->stream_comm));
  return 0;
}

// Vector add: out = a + b
__global__ void add2_vec_bell(const float* __restrict__ a,
                              const float* __restrict__ b,
                              float* __restrict__ out,
                              int n) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) out[t] = a[t] + b[t];
}

int parbell_spmv(ParBELL *P) {
  const float alpha = 1.0f;
  // Lazily allocate split outputs and descriptors
  if (!P->d_y_aii) CHECK_CUDA(cudaMalloc((void**)&P->d_y_aii, (size_t)(P->mb_local * P->bdim) * sizeof(float)));
  if (P->nb_ghost > 0 && !P->d_y_aij) CHECK_CUDA(cudaMalloc((void**)&P->d_y_aij, (size_t)(P->mb_local * P->bdim) * sizeof(float)));
  if (!P->dnY_aii) {
    int64_t mY = (int64_t)P->mb_local * P->bdim; int64_t n = 1;
    if (cusparseCreateDnMat((cusparseDnMatDescr_t*)&P->dnY_aii, mY, n, mY, (void*)P->d_y_aii, CUDA_R_32F, CUSPARSE_ORDER_COL) != CUSPARSE_STATUS_SUCCESS) return -1;
  }
  if (P->nb_ghost > 0 && !P->dnY_aij) {
    int64_t mY = (int64_t)P->mb_local * P->bdim; int64_t n = 1;
    if (cusparseCreateDnMat((cusparseDnMatDescr_t*)&P->dnY_aij, mY, n, mY, (void*)P->d_y_aij, CUDA_R_32F, CUSPARSE_ORDER_COL) != CUSPARSE_STATUS_SUCCESS) return -1;
  }
  // Zero outputs
  CHECK_CUDA(cudaMemsetAsync(P->d_y_aii, 0, (size_t)(P->mb_local * P->bdim) * sizeof(float), P->stream_comp));
  if (P->nb_ghost > 0)
    CHECK_CUDA(cudaMemsetAsync(P->d_y_aij, 0, (size_t)(P->mb_local * P->bdim) * sizeof(float), P->stream_comp2));

  // Launch Aii on stream_comp
  if (P->matAii) {
    float beta0 = 0.0f;
    cusparseStatus_t s = cusparseSpMM(P->sp_handle,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      &alpha,
                                      (cusparseSpMatDescr_t)P->matAii,
                                      (cusparseDnMatDescr_t)P->dnX_local,
                                      &beta0,
                                      (cusparseDnMatDescr_t)P->dnY_aii,
                                      CUDA_R_32F,
                                      CUSPARSE_SPMM_ALG_DEFAULT,
                                      P->d_workspace_aii);
    if (s != CUSPARSE_STATUS_SUCCESS) { debug_logf(P->pe, "SpMM(Aii) failed %d", (int)s); return -1; }
  }
  // Launch Aij on stream_comp2 after halo event
  CHECK_CUDA(cudaStreamWaitEvent(P->stream_comp2, P->evt_halo_done, 0));
  if (P->matAij && P->Aij_ell_cols > 0 && P->nb_ghost > 0) {
    float beta0 = 0.0f;
    cusparseStatus_t s = cusparseSpMM(P->sp_handle_aij,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                      &alpha,
                                      (cusparseSpMatDescr_t)P->matAij,
                                      (cusparseDnMatDescr_t)P->dnX_ghost,
                                      &beta0,
                                      (cusparseDnMatDescr_t)P->dnY_aij,
                                      CUDA_R_32F,
                                      CUSPARSE_SPMM_ALG_DEFAULT,
                                      P->d_workspace_aij);
    if (s != CUSPARSE_STATUS_SUCCESS) { debug_logf(P->pe, "SpMM(Aij) failed %d", (int)s); return -1; }
  }

  // Synchronize and sum into y
  CHECK_CUDA(cudaStreamSynchronize(P->stream_comp));
  CHECK_CUDA(cudaStreamSynchronize(P->stream_comp2));
  int m = P->mb_local * P->bdim;
  if (P->nb_ghost > 0) {
    int threads = 256; int blocks = (m + threads - 1)/threads;
    add2_vec_bell<<<blocks, threads, 0, P->stream_comp>>>(P->d_y_aii, P->d_y_aij, P->d_y, m);
    CHECK_CUDA(cudaStreamSynchronize(P->stream_comp));
  } else {
    CHECK_CUDA(cudaMemcpyAsync(P->d_y, P->d_y_aii, (size_t)m * sizeof(float), cudaMemcpyDeviceToDevice, P->stream_comp));
    CHECK_CUDA(cudaStreamSynchronize(P->stream_comp));
  }
  return 0;
}

void parbell_print_vec(const ParBELL *P, const char *name, const float *dptr,
                      int n) {
  if (n <= 0) {
    debug_logf(P->pe, "%s: (empty)", name);
    return;
  }
  float *h = (float *)std::malloc(n * sizeof(float));
  cudaMemcpy(h, dptr, n * sizeof(float), cudaMemcpyDeviceToHost);
  // Build a single line to reduce interleaving
  std::string line;
  line.reserve(16 + n * 6);
  line += name;
  line += ":";
  char buf[64];
  for (int i = 0; i < n; ++i) {
    int len = snprintf(buf, sizeof(buf), " %.1f", h[i]);
    line.append(buf, buf + len);
  }
  debug_logf(P->pe, "%s", line.c_str());
  std::free(h);
}

void parbell_free(ParBELL *P) {
  if (!P)
    return;
  // device
  if (P->d_Aii_colind)
    cudaFree(P->d_Aii_colind);
  if (P->d_Aii_val)
    cudaFree(P->d_Aii_val);
  if (P->d_Aij_colind)
    cudaFree(P->d_Aij_colind);
  if (P->d_Aij_val)
    cudaFree(P->d_Aij_val);
  if (P->d_x_local)
    cudaFree(P->d_x_local);
  if (P->d_x_ghost)
    cudaFree(P->d_x_ghost);
  if (P->d_y)
    cudaFree(P->d_y);
  if (P->d_sendbuf)
    cudaFree(P->d_sendbuf);
  if (P->d_recvbuf)
    cudaFree(P->d_recvbuf);

  // host
  if (P->h_col_map_offd)
    free(P->h_col_map_offd);
  // Comm plan resources
  comm_plan_free(&P->plan);

  // handles and modern API descriptors
  if (P->matAii)
    cusparseDestroySpMat((cusparseSpMatDescr_t)P->matAii);
  if (P->matAij)
    cusparseDestroySpMat((cusparseSpMatDescr_t)P->matAij);
  if (P->dnX_local)
    cusparseDestroyDnMat((cusparseDnMatDescr_t)P->dnX_local);
  if (P->dnX_ghost)
    cusparseDestroyDnMat((cusparseDnMatDescr_t)P->dnX_ghost);
  if (P->dnY)
    cusparseDestroyDnMat((cusparseDnMatDescr_t)P->dnY);
  if (P->dnY_aii)
    cusparseDestroyDnMat((cusparseDnMatDescr_t)P->dnY_aii);
  if (P->dnY_aij)
    cusparseDestroyDnMat((cusparseDnMatDescr_t)P->dnY_aij);
  if (P->d_workspace_aii)
    cudaFree(P->d_workspace_aii);
  if (P->d_workspace_aij)
    cudaFree(P->d_workspace_aij);
  if (P->sp_handle)
    cusparseDestroy(P->sp_handle);
  if (P->sp_handle_aij)
    cusparseDestroy(P->sp_handle_aij);
  if (P->stream_comm)
    cudaStreamDestroy(P->stream_comm);
  if (P->stream_comp)
    cudaStreamDestroy(P->stream_comp);
  if (P->stream_comp2)
    cudaStreamDestroy(P->stream_comp2);
  if (P->evt_halo_done)
    cudaEventDestroy(P->evt_halo_done);
  if (P->d_y_aii)
    cudaFree(P->d_y_aii);
  if (P->d_y_aij)
    cudaFree(P->d_y_aij);

  memset(P, 0, sizeof(*P));
}
