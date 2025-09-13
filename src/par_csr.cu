#include "par_csr.h"
#include "utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------- kernels: pack/unpack scalars ----------------
__global__ void pack_f32(const float *__restrict__ x_local,
                         float *__restrict__ sendbuf,
                         const int *__restrict__ ids, int n) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n)
    sendbuf[t] = x_local[ids[t]];
}

__global__ void unpack_f32(const float *__restrict__ recvbuf,
                           float *__restrict__ x_ghost,
                           const int *__restrict__ ids, int n) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n)
    x_ghost[ids[t]] = recvbuf[t];
}

// Vector add: out = a + b
__global__ void add2_vec_csr(const float *__restrict__ a, const float *__restrict__ b,
                             float *__restrict__ out, int n) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n)
    out[t] = a[t] + b[t];
}

// ---------------- helpers ----------------
int upload_csr(int m, int nnz, const int *h_rowptr, const int *h_colind,
               const float *h_val, int **d_rowptr, int **d_colind,
               float **d_val) {
  CHECK_CUDA(cudaMalloc((void **)d_rowptr, (size_t)(m + 1) * sizeof(int)));
  CHECK_CUDA(
      cudaMalloc((void **)d_colind, (size_t)(nnz > 0 ? nnz : 1) * sizeof(int)));
  CHECK_CUDA(
      cudaMalloc((void **)d_val, (size_t)(nnz > 0 ? nnz : 1) * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(*d_rowptr, h_rowptr, (size_t)(m + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
  if (nnz > 0) {
    CHECK_CUDA(cudaMemcpy(*d_colind, h_colind, (size_t)nnz * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(*d_val, h_val, (size_t)nnz * sizeof(float),
                          cudaMemcpyHostToDevice));
  }
  return 0;
}

static int create_dnvec(float *ptr, int n, void **out) {
  cusparseStatus_t s = cusparseCreateDnVec((cusparseDnVecDescr_t *)out,
                                           (int64_t)n, (void *)ptr, CUDA_R_32F);
  return (s == CUSPARSE_STATUS_SUCCESS) ? 0 : -1;
}

// Use a fixed cuSPARSE SpMV algorithm for determinism and performance
static inline cusparseSpMVAlg_t spmv_alg() { return CUSPARSE_SPMV_CSR_ALG1; }

int parcsr_create_cusparse_descriptors(ParCSR *P, int nnzAii, int nnzAij) {
  // CSR sparse matrices
  if (nnzAii > 0) {
    int64_t m = P->m_local, k = P->n_local;
    cusparseStatus_t s = cusparseCreateCsr(
        (cusparseSpMatDescr_t *)&P->matAii, m, k, (int64_t)nnzAii,
        P->d_Aii_rowptr, P->d_Aii_colind, P->d_Aii_val, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    if (s != CUSPARSE_STATUS_SUCCESS)
      return -1;
  }
  if (nnzAij > 0 && P->n_ghost > 0) {
    int64_t m = P->m_local, k = P->n_ghost;
    cusparseStatus_t s = cusparseCreateCsr(
        (cusparseSpMatDescr_t *)&P->matAij, m, k, (int64_t)nnzAij,
        P->d_Aij_rowptr, P->d_Aij_colind, P->d_Aij_val, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    if (s != CUSPARSE_STATUS_SUCCESS)
      return -1;
  }

  // Dense vectors
  if (create_dnvec(P->d_x_local, P->n_local, &P->dnX_local))
    return -1;
  if (create_dnvec(P->d_x_ghost, (P->n_ghost > 0 ? P->n_ghost : 1),
                   &P->dnX_ghost))
    return -1;
  if (create_dnvec(P->d_y, P->m_local, &P->dnY))
    return -1;
  // Split-parallel buffers created lazily on first use
  P->d_y_aii = nullptr;
  P->d_y_aij = nullptr;
  P->dnY_aii = nullptr;
  P->dnY_aij = nullptr;

  // Per-matrix workspace queries and allocations (default)
  P->workspace_size_aii = 0;
  P->workspace_size_aij = 0;
  P->d_workspace_aii = nullptr;
  P->d_workspace_aij = nullptr;
  if (P->matAii) {
    const float alpha = 1.0f, beta = 0.0f;
    size_t sz = 0;
    if (cusparseSpMV_bufferSize(P->sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, (cusparseSpMatDescr_t)P->matAii,
                                (cusparseDnVecDescr_t)P->dnX_local, &beta,
                                (cusparseDnVecDescr_t)P->dnY, CUDA_R_32F,
                                spmv_alg(), &sz) != CUSPARSE_STATUS_SUCCESS)
      return -1;
    P->workspace_size_aii = sz;
    if (sz > 0)
      CHECK_CUDA(cudaMalloc(&P->d_workspace_aii, sz));
  }
  if (P->matAij && P->n_ghost > 0) {
    const float alpha = 1.0f, beta = 1.0f;
    size_t sz = 0;
    if (cusparseSpMV_bufferSize(
            P->sp_handle_aij, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
            (cusparseSpMatDescr_t)P->matAij, (cusparseDnVecDescr_t)P->dnX_ghost,
            &beta, (cusparseDnVecDescr_t)P->dnY, CUDA_R_32F, spmv_alg(),
            &sz) != CUSPARSE_STATUS_SUCCESS)
      return -1;
    P->workspace_size_aij = sz;
    if (sz > 0)
      CHECK_CUDA(cudaMalloc(&P->d_workspace_aij, sz));
  }
  // Optional preprocessing to accelerate repeated SpMV with same sparsity
  if (P->matAii) {
    const float alpha = 1.0f, beta = 0.0f;
    if (cusparseSpMV_preprocess(
            P->sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
            (cusparseSpMatDescr_t)P->matAii, (cusparseDnVecDescr_t)P->dnX_local,
            &beta, (cusparseDnVecDescr_t)P->dnY, CUDA_R_32F, spmv_alg(),
            P->d_workspace_aii) != CUSPARSE_STATUS_SUCCESS)
      return -1;
  }
  if (P->matAij && P->n_ghost > 0) {
    const float alpha = 1.0f, beta = 1.0f;
    if (cusparseSpMV_preprocess(
            P->sp_handle_aij, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
            (cusparseSpMatDescr_t)P->matAij, (cusparseDnVecDescr_t)P->dnX_ghost,
            &beta, (cusparseDnVecDescr_t)P->dnY, CUDA_R_32F, spmv_alg(),
            P->d_workspace_aij) != CUSPARSE_STATUS_SUCCESS)
      return -1;
  }
  return 0;
}

// ---------------- API impl ----------------

int parcsr_init(ParCSR *P, int pe, int npes, int m_local, int n_local,
                gblk *col_starts, ncclComm_t nccl_comm) {
  if (!P)
    return -1;
  std::memset(P, 0, sizeof(*P));
  P->pe = pe;
  P->npes = npes;
  P->m_local = m_local;
  P->n_local = n_local;
  P->n_ghost = 0;
  P->h_col_starts = col_starts;
  P->nccl = nccl_comm;

  CHECK_CUDA(cudaStreamCreate(&P->stream_comm));
  CHECK_CUDA(cudaStreamCreate(&P->stream_comp));
  CHECK_CUSPARSE(cusparseCreate(&P->sp_handle));
  CHECK_CUSPARSE(cusparseSetStream(P->sp_handle, P->stream_comp));
  // Second compute stream + handle for split-parallel SpMV
  CHECK_CUDA(cudaStreamCreate(&P->stream_comp2));
  CHECK_CUSPARSE(cusparseCreate(&P->sp_handle_aij));
  CHECK_CUSPARSE(cusparseSetStream(P->sp_handle_aij, P->stream_comp2));
  // Halo completion event
  CHECK_CUDA(cudaEventCreateWithFlags(&P->evt_halo_done, cudaEventDisableTiming));
  return 0;
}

int parcsr_build_comm_plan_from_colmap(ParCSR *P, MPI_Comm comm) {
  // host ghost list already in P->h_col_map_offd / n_ghost_host
  int rc =
      comm_plan_build_from_colmap(&P->plan, P->pe, P->npes, P->n_ghost_host,
                                  P->h_col_map_offd, P->h_col_starts, comm);
  if (rc != 0)
    return rc;
  int total_recv = comm_plan_total_recv(&P->plan);
  int total_send = comm_plan_total_send(&P->plan);
  if (total_send > 0)
    CHECK_CUDA(
        cudaMalloc((void **)&P->d_sendbuf, (size_t)total_send * sizeof(float)));
  if (total_recv > 0)
    CHECK_CUDA(
        cudaMalloc((void **)&P->d_recvbuf, (size_t)total_recv * sizeof(float)));
  return 0;
}

int parcsr_halo_x(ParCSR *P) {
  int total_send = comm_plan_total_send(&P->plan);
  int total_recv = comm_plan_total_recv(&P->plan);
  int threads = 128;

  // Pack by neighbor contiguous segments
  for (int a = 0, off = 0; a < P->plan.n_neighbors; ++a) {
    int cnt = P->plan.send_counts[a];
    if (cnt > 0) {
      pack_f32<<<(cnt + threads - 1) / threads, threads, 0, P->stream_comm>>>(
          P->d_x_local, P->d_sendbuf + off,
          P->plan.d_send_ids + P->plan.send_offsets[a], cnt);
    }
    off += cnt;
  }

  if (env_flag("HALO_USE_MPI", 0)) {
    // Debug fallback: exchange via MPI on host buffers (avoids NCCL)
    int total_send = comm_plan_total_send(&P->plan);
    int total_recv = comm_plan_total_recv(&P->plan);
    std::vector<float> h_send(total_send);
    std::vector<float> h_recv(total_recv);
    CHECK_CUDA(cudaMemcpyAsync(h_send.data(), P->d_sendbuf,
                               (size_t)total_send * sizeof(float),
                               cudaMemcpyDeviceToHost, P->stream_comm));
    CHECK_CUDA(cudaStreamSynchronize(P->stream_comm));
    std::vector<MPI_Request> reqs;
    reqs.reserve((size_t)P->plan.n_neighbors * 2);
    // Post all Irecv
    for (int a = 0, off = 0; a < P->plan.n_neighbors; ++a) {
      int peer = P->plan.neighbors[a];
      int cnt = P->plan.recv_counts[a];
      if (cnt > 0) {
        MPI_Request r;
        CHECK_MPI(MPI_Irecv(h_recv.data() + off, cnt, MPI_FLOAT, peer, 100,
                            MPI_COMM_WORLD, &r));
        reqs.push_back(r);
      }
      off += cnt;
    }
    // Post all Isend
    for (int a = 0, off = 0; a < P->plan.n_neighbors; ++a) {
      int peer = P->plan.neighbors[a];
      int cnt = P->plan.send_counts[a];
      if (cnt > 0) {
        MPI_Request r;
        CHECK_MPI(MPI_Isend(h_send.data() + off, cnt, MPI_FLOAT, peer, 100,
                            MPI_COMM_WORLD, &r));
        reqs.push_back(r);
      }
      off += cnt;
    }
    if (!reqs.empty())
      CHECK_MPI(
          MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE));
    CHECK_CUDA(cudaMemcpyAsync(P->d_recvbuf, h_recv.data(),
                               (size_t)total_recv * sizeof(float),
                               cudaMemcpyHostToDevice, P->stream_comm));
  } else {
    // NCCL exchange grouped: post all recvs then sends
    CHECK_NCCL(ncclGroupStart());
    int off_r = 0, off_s = 0;
    for (int a = 0; a < P->plan.n_neighbors; ++a) {
      int peer = P->plan.neighbors[a];
      int cnt_r = P->plan.recv_counts[a];
      int cnt_s = P->plan.send_counts[a];
      if (cnt_r > 0)
        CHECK_NCCL(ncclRecv(P->d_recvbuf + off_r, cnt_r, ncclFloat, peer,
                            P->nccl, P->stream_comm));
      if (cnt_s > 0)
        CHECK_NCCL(ncclSend(P->d_sendbuf + off_s, cnt_s, ncclFloat, peer,
                            P->nccl, P->stream_comm));
      off_r += cnt_r;
      off_s += cnt_s;
    }
    CHECK_NCCL(ncclGroupEnd());
  }

  // Unpack recvbuf into x_ghost by neighbor segments
  for (int a = 0, off = 0; a < P->plan.n_neighbors; ++a) {
    int cnt = P->plan.recv_counts[a];
    if (cnt > 0) {
      unpack_f32<<<(cnt + threads - 1) / threads, threads, 0, P->stream_comm>>>(
          P->d_recvbuf + off, P->d_x_ghost,
          P->plan.d_recv_ids + P->plan.recv_offsets[a], cnt);
    }
    off += cnt;
  }

  // Record halo completion; do not synchronize here to enable overlap
  CHECK_CUDA(cudaEventRecord(P->evt_halo_done, P->stream_comm));
  return 0;
}

int parcsr_spmv(ParCSR *P) {
  const float alpha = 1.0f;
  {
    // Allocate split outputs and descriptors lazily
    if (!P->d_y_aii)
      CHECK_CUDA(
          cudaMalloc((void **)&P->d_y_aii, (size_t)P->m_local * sizeof(float)));
    if (!P->d_y_aij)
      CHECK_CUDA(
          cudaMalloc((void **)&P->d_y_aij, (size_t)P->m_local * sizeof(float)));
    if (!P->dnY_aii)
      if (create_dnvec(P->d_y_aii, P->m_local, &P->dnY_aii))
        return -1;
    if (!P->dnY_aij)
      if (create_dnvec(P->d_y_aij, P->m_local, &P->dnY_aij))
        return -1;
    // Zero outputs
    CHECK_CUDA(cudaMemsetAsync(
        P->d_y_aii, 0, (size_t)P->m_local * sizeof(float), P->stream_comp));
    if (P->matAij && P->n_ghost > 0)
      CHECK_CUDA(cudaMemsetAsync(
          P->d_y_aij, 0, (size_t)P->m_local * sizeof(float), P->stream_comp2));
    // Launch Aii on stream_comp
    if (P->matAii) {
      float beta0 = 0.0f;
      cusparseStatus_t s = cusparseSpMV(
          P->sp_handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
          (cusparseSpMatDescr_t)P->matAii, (cusparseDnVecDescr_t)P->dnX_local,
          &beta0, (cusparseDnVecDescr_t)P->dnY_aii, CUDA_R_32F, spmv_alg(),
          P->d_workspace_aii);
      if (s != CUSPARSE_STATUS_SUCCESS)
        return -1;
    }
    // Launch Aij on stream_comp2 after halo event
    CHECK_CUDA(cudaStreamWaitEvent(P->stream_comp2, P->evt_halo_done, 0));
    if (P->matAij && P->n_ghost > 0) {
      float beta0 = 0.0f;
      cusparseStatus_t s = cusparseSpMV(
          P->sp_handle_aij, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
          (cusparseSpMatDescr_t)P->matAij, (cusparseDnVecDescr_t)P->dnX_ghost,
          &beta0, (cusparseDnVecDescr_t)P->dnY_aij, CUDA_R_32F, spmv_alg(),
          P->d_workspace_aij);
      if (s != CUSPARSE_STATUS_SUCCESS)
        return -1;
    }
    // Wait for both, then sum into y
    CHECK_CUDA(cudaStreamSynchronize(P->stream_comp));
    CHECK_CUDA(cudaStreamSynchronize(P->stream_comp2));
    if (P->n_ghost > 0) {
      int threads = 256;
      int blocks = (P->m_local + threads - 1) / threads;
      add2_vec_csr<<<blocks, threads, 0, P->stream_comp>>>(P->d_y_aii, P->d_y_aij, P->d_y,
                                                           P->m_local);
      CHECK_CUDA(cudaStreamSynchronize(P->stream_comp));
    } else {
      // No off-diagonal contribution; just copy Aii result into y
      CHECK_CUDA(cudaMemcpyAsync(P->d_y, P->d_y_aii, (size_t)P->m_local * sizeof(float),
                                 cudaMemcpyDeviceToDevice, P->stream_comp));
      CHECK_CUDA(cudaStreamSynchronize(P->stream_comp));
    }
  }
  return 0;
}

void parcsr_print_vec(const ParCSR *P, const char *name, const float *dptr,
                      int n) {
  if (n <= 0) {
    debug_logf(P->pe, "%s: (empty)", name);
    return;
  }
  float *h = (float *)std::malloc((size_t)n * sizeof(float));
  cudaMemcpy(h, dptr, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);
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

void parcsr_free(ParCSR *P) {
  if (!P)
    return;
  // device arrays
  if (P->d_Aii_rowptr)
    cudaFree(P->d_Aii_rowptr);
  if (P->d_Aii_colind)
    cudaFree(P->d_Aii_colind);
  if (P->d_Aii_val)
    cudaFree(P->d_Aii_val);
  if (P->d_Aij_rowptr)
    cudaFree(P->d_Aij_rowptr);
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

  // comm plan
  comm_plan_free(&P->plan);

  // descriptors and handles
  if (P->matAii)
    cusparseDestroySpMat((cusparseSpMatDescr_t)P->matAii);
  if (P->matAij)
    cusparseDestroySpMat((cusparseSpMatDescr_t)P->matAij);
  if (P->dnX_local)
    cusparseDestroyDnVec((cusparseDnVecDescr_t)P->dnX_local);
  if (P->dnX_ghost)
    cusparseDestroyDnVec((cusparseDnVecDescr_t)P->dnX_ghost);
  if (P->dnY)
    cusparseDestroyDnVec((cusparseDnVecDescr_t)P->dnY);
  if (P->dnY_aii)
    cusparseDestroyDnVec((cusparseDnVecDescr_t)P->dnY_aii);
  if (P->dnY_aij)
    cusparseDestroyDnVec((cusparseDnVecDescr_t)P->dnY_aij);
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

  // host arrays
  if (P->h_col_map_offd)
    free(P->h_col_map_offd);

  std::memset(P, 0, sizeof(*P));
}
