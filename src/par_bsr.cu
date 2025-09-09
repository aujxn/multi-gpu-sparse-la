#include "par_bsr.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <mpi.h>
#include <nccl.h>
#include <vector>

#define CHECK_CUDA(x)                                                          \
  do {                                                                         \
    cudaError_t e = (x);                                                       \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,                  \
              cudaGetErrorString(e));                                          \
      return -1;                                                               \
    }                                                                          \
  } while (0)
#define CHECK_CUSPARSE(x)                                                      \
  do {                                                                         \
    cusparseStatus_t s = (x);                                                  \
    if (s != CUSPARSE_STATUS_SUCCESS) {                                        \
      fprintf(stderr, "cuSPARSE %s:%d: %d\n", __FILE__, __LINE__, (int)s);     \
      return -1;                                                               \
    }                                                                          \
  } while (0)
#define CHECK_NCCL(x)                                                          \
  do {                                                                         \
    ncclResult_t n = (x);                                                      \
    if (n != ncclSuccess) {                                                    \
      fprintf(stderr, "NCCL %s:%d: %s\n", __FILE__, __LINE__,                  \
              ncclGetErrorString(n));                                          \
      return -1;                                                               \
    }                                                                          \
  } while (0)

static inline int owner_of_block(gblk g, const gblk *col_starts, int npes) {
  // contiguous ownership; binary search would be nicer for big npes
  for (int p = 0; p < npes; ++p)
    if (g >= col_starts[p] && g < col_starts[p + 1])
      return p;
  return -1;
}

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

int parbsr_init(ParBSR *P, int pe, int npes, int mb_local, int bdim,
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
  CHECK_CUSPARSE(cusparseCreate(&P->sp_handle));
  CHECK_CUSPARSE(cusparseSetStream(P->sp_handle, P->stream_comp));

  // matrix descriptors
  cusparseMatDescr_t d1 = nullptr, d2 = nullptr;
  CHECK_CUSPARSE(cusparseCreateMatDescr(&d1));
  CHECK_CUSPARSE(cusparseSetMatType(d1, CUSPARSE_MATRIX_TYPE_GENERAL));
  CHECK_CUSPARSE(cusparseSetMatIndexBase(d1, CUSPARSE_INDEX_BASE_ZERO));
  CHECK_CUSPARSE(cusparseCreateMatDescr(&d2));
  CHECK_CUSPARSE(cusparseSetMatType(d2, CUSPARSE_MATRIX_TYPE_GENERAL));
  CHECK_CUSPARSE(cusparseSetMatIndexBase(d2, CUSPARSE_INDEX_BASE_ZERO));
  P->descrAii = d1;
  P->descrAij = d2;

  return 0;
}

static int upload_bsr(int mb, int nnzb, int bdim, const int *h_rowptr,
                      const int *h_colind, const float *h_val, int **d_rowptr,
                      int **d_colind, float **d_val) {
  CHECK_CUDA(cudaMalloc((void **)d_rowptr, (mb + 1) * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(*d_rowptr, h_rowptr, (mb + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
  if (nnzb > 0) {
    CHECK_CUDA(cudaMalloc((void **)d_colind, nnzb * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)d_val, nnzb * bdim * bdim * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(*d_colind, h_colind, nnzb * sizeof(int),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(*d_val, h_val, nnzb * bdim * bdim * sizeof(float),
                          cudaMemcpyHostToDevice));
  } else {
    *d_colind = nullptr;
    *d_val = nullptr;
  }
  return 0;
}

int parbsr_build_toy(ParBSR *P) {
  // A_ii: two diagonal blocks 2*I
  int h_rp_ii[3] = {0, 1, 2};
  int h_ci_ii[2] = {0, 1};
  float h_val_ii[2 * P->bdim * P->bdim];
  for (int b = 0; b < 2; ++b)
    for (int r = 0; r < P->bdim; ++r)
      for (int c = 0; c < P->bdim; ++c)
        h_val_ii[b * P->bdim * P->bdim + r * P->bdim + c] =
            (r == c) ? 2.0f : 0.0f;

  P->Aii_nnzb = 2;
  if (upload_bsr(P->mb_local, P->Aii_nnzb, P->bdim, h_rp_ii, h_ci_ii, h_val_ii,
                 &P->d_Aii_rowptr, &P->d_Aii_colind, &P->d_Aii_val))
    return -1;

  // A_ij: if right neighbor exists, one ones-block in last block-row → ghost
  // col 0
  int has_right = (P->pe + 1 < P->npes) ? 1 : 0;
  int h_rp_ij[3];
  int h_ci_ij_[1];
  int *h_ci_ij = nullptr;
  float h_val_ij_[16];
  float *h_val_ij = nullptr;

  if (has_right) {
    h_rp_ij[0] = 0;
    h_rp_ij[1] = 0;
    h_rp_ij[2] = 1;
    h_ci_ij = h_ci_ij_;
    h_ci_ij[0] = 0;
    h_val_ij = h_val_ij_;
    for (int i = 0; i < P->bdim * P->bdim; ++i)
      h_val_ij[i] = 1.0f;
    P->Aij_nnzb = 1;
    P->nb_ghost_host = 1;
  } else {
    h_rp_ij[0] = 0;
    h_rp_ij[1] = 0;
    h_rp_ij[2] = 0;
    P->Aij_nnzb = 0;
    P->nb_ghost_host = 0;
  }
  P->nb_ghost = P->nb_ghost_host;

  if (upload_bsr(P->mb_local, P->Aij_nnzb, P->bdim, h_rp_ij, h_ci_ij, h_val_ij,
                 &P->d_Aij_rowptr, &P->d_Aij_colind, &P->d_Aij_val))
    return -1;

  // Ghost map: first block of right neighbor as global id
  if (has_right) {
    P->h_col_map_offd = (gblk *)std::malloc(sizeof(gblk));
    P->h_col_map_offd[0] = P->h_col_starts[P->pe + 1]; // first block of right
  } else {
    P->h_col_map_offd = nullptr;
  }

  // Allocate vectors
  int n_local = P->mb_local * P->bdim;
  int n_ghost = P->nb_ghost * P->bdim;
  CHECK_CUDA(cudaMalloc((void **)&P->d_x_local, n_local * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&P->d_x_ghost,
                        (n_ghost > 0 ? n_ghost : 1) * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&P->d_y, n_local * sizeof(float)));

  // Initialize x_local = 100*pe + [0..), zero x_ghost,y
  float *htmp = (float *)std::malloc(n_local * sizeof(float));
  for (int i = 0; i < n_local; ++i)
    htmp[i] = 100.0f * P->pe + (float)i;
  CHECK_CUDA(cudaMemcpy(P->d_x_local, htmp, n_local * sizeof(float),
                        cudaMemcpyHostToDevice));
  std::free(htmp);
  CHECK_CUDA(
      cudaMemset(P->d_x_ghost, 0, (n_ghost > 0 ? n_ghost : 1) * sizeof(float)));
  CHECK_CUDA(cudaMemset(P->d_y, 0, n_local * sizeof(float)));
  return 0;
}

int parbsr_build_comm_plan_ring(ParBSR *P) {
  // Receive from right neighbor: all ghost ids belong to right owner
  int has_right = (P->pe + 1 < P->npes) ? 1 : 0;
  int has_left = (P->pe - 1 >= 0) ? 1 : 0;

  // how many ghost BLOCKS we receive from right (toy = 1 if has_right)
  int nrecvb = has_right ? 1 : 0;

  P->n_neighbors = has_right + has_left;
  if (P->n_neighbors == 0) {
    P->neighbors = P->recv_counts = P->recv_offsets = P->send_counts =
        P->send_offsets = nullptr;
    P->h_recv_ghost_ids = P->h_send_local_block_ids = nullptr;
    P->d_recv_ghost_ids = P->d_send_local_block_ids = nullptr;
    P->d_sendbuf = P->d_recvbuf = nullptr;
    return 0;
  }

  P->neighbors = (int *)std::malloc(P->n_neighbors * sizeof(int));
  P->recv_counts = (int *)std::calloc(P->n_neighbors, sizeof(int));
  P->recv_offsets = (int *)std::malloc(P->n_neighbors * sizeof(int));
  P->send_counts = (int *)std::calloc(P->n_neighbors, sizeof(int));
  P->send_offsets = (int *)std::malloc(P->n_neighbors * sizeof(int));

  int idx = 0;
  if (has_right)
    P->neighbors[idx++] = P->pe + 1;
  if (has_left)
    P->neighbors[idx++] = P->pe - 1;

  // RECV: from right neighbor only (matches toy)
  for (int a = 0; a < P->n_neighbors; ++a)
    if (P->neighbors[a] == P->pe + 1)
      P->recv_counts[a] = nrecvb;

  int total_recv = 0;
  for (int a = 0; a < P->n_neighbors; ++a) {
    P->recv_offsets[a] = total_recv;
    total_recv += P->recv_counts[a];
  }

  if (total_recv > 0) {
    P->h_recv_ghost_ids = (int *)std::malloc(total_recv * sizeof(int));
    for (int i = 0; i < total_recv; ++i)
      P->h_recv_ghost_ids[i] = i; // ghost ids are 0..(nrecvb-1)
  } else
    P->h_recv_ghost_ids = nullptr;

  // SEND: to left neighbor what THEY need (toy = 1 if we have a left neighbor)
  for (int a = 0; a < P->n_neighbors; ++a)
    if (P->neighbors[a] == P->pe - 1)
      P->send_counts[a] = has_left ? 1 : 0;

  int total_send = 0;
  for (int a = 0; a < P->n_neighbors; ++a) {
    P->send_offsets[a] = total_send;
    total_send += P->send_counts[a];
  }

  if (total_send > 0) {
    P->h_send_local_block_ids = (int *)std::malloc(total_send * sizeof(int));
    for (int i = 0; i < total_send; ++i)
      P->h_send_local_block_ids[i] = 0; // toy: first local block
  } else
    P->h_send_local_block_ids = nullptr;

  // Device mirrors + scratch
  CHECK_CUDA(
      cudaMalloc((void **)&P->d_recv_ghost_ids, total_recv * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void **)&P->d_send_local_block_ids,
                        total_send * sizeof(int)));
  if (total_recv > 0)
    CHECK_CUDA(cudaMemcpy(P->d_recv_ghost_ids, P->h_recv_ghost_ids,
                          total_recv * sizeof(int), cudaMemcpyHostToDevice));
  if (total_send > 0)
    CHECK_CUDA(cudaMemcpy(P->d_send_local_block_ids, P->h_send_local_block_ids,
                          total_send * sizeof(int), cudaMemcpyHostToDevice));

  CHECK_CUDA(
      cudaMalloc((void **)&P->d_sendbuf, total_send * P->bdim * sizeof(float)));
  CHECK_CUDA(
      cudaMalloc((void **)&P->d_recvbuf, total_recv * P->bdim * sizeof(float)));
  return 0;
}

int parbsr_build_comm_plan_from_colmap(ParBSR *P, MPI_Comm comm) {
  // ---- gather who we need (owners) and group our ghost ids per owner ----
  std::vector<int> owners;
  owners.reserve(P->nb_ghost_host);
  std::vector<int> recv_counts_vec(P->npes, 0);
  std::vector<std::vector<gblk>> req_gblks(
      P->npes); // what we request from peer
  std::vector<std::vector<int>> req_ghostids(
      P->npes); // matching ghost ids (our order)

  for (int g = 0; g < P->nb_ghost_host; ++g) {
    gblk G = P->h_col_map_offd[g];
    int owner = owner_of_block(G, P->h_col_starts, P->npes);
    if (owner == P->pe)
      continue; // shouldn't happen in offd
    if (recv_counts_vec[owner] == 0)
      owners.push_back(owner);
    recv_counts_vec[owner] += 1;
    req_gblks[owner].push_back(G);
    req_ghostids[owner].push_back(g);
  }

  // ---- Alltoall counts: tell each owner how many blocks we need from them
  // ----
  std::vector<int> send_counts_req(P->npes, 0);
  for (int p = 0; p < P->npes; ++p)
    send_counts_req[p] = recv_counts_vec[p];

  std::vector<int> recv_counts_req(P->npes, 0);
  MPI_Alltoall(send_counts_req.data(), 1, MPI_INT, recv_counts_req.data(), 1,
               MPI_INT, comm);

  // recv_counts_req[p] = how many block ids peer p requests from *us*.
  // This defines our SEND side.

  // ---- Compute displacements and flatten our outgoing request lists ----
  std::vector<int> sdispl(P->npes, 0), rdispl(P->npes, 0);
  int send_total = 0, recv_total = 0;
  for (int p = 1; p < P->npes; ++p)
    sdispl[p] = sdispl[p - 1] + send_counts_req[p - 1];
  for (int p = 1; p < P->npes; ++p)
    rdispl[p] = rdispl[p - 1] + recv_counts_req[p - 1];
  for (int p = 0; p < P->npes; ++p) {
    send_total += send_counts_req[p];
    recv_total += recv_counts_req[p];
  }

  std::vector<gblk> send_gblks(send_total);
  // On receiver side we also need to remember the order of ghost ids for
  // unpack:
  std::vector<int> send_ghostids(send_total);
  // Fill by peer in ascending order to match displacements
  for (int p = 0; p < P->npes; ++p) {
    int off = sdispl[p];
    for (size_t i = 0; i < req_gblks[p].size(); ++i) {
      send_gblks[off + (int)i] = req_gblks[p][i];
      send_ghostids[off + (int)i] = req_ghostids[p][i];
    }
  }

  // ---- Exchange the requested global block ids ----
  std::vector<gblk> recv_gblks(recv_total);
  MPI_Datatype mpi_gblk = (sizeof(gblk) == 8) ? MPI_LONG_LONG : MPI_INT;
  MPI_Alltoallv(send_gblks.data(), send_counts_req.data(), sdispl.data(),
                mpi_gblk, recv_gblks.data(), recv_counts_req.data(),
                rdispl.data(), mpi_gblk, comm);

  // ---- Build neighbor list = union of owners we RECV from and peers we SEND
  // to ----
  std::vector<char> neighbor_mask(P->npes, 0);
  for (int p = 0; p < P->npes; ++p)
    if (send_counts_req[p] > 0)
      neighbor_mask[p] = 1; // we recv from p
  for (int p = 0; p < P->npes; ++p)
    if (recv_counts_req[p] > 0)
      neighbor_mask[p] = 1; // we send to p
  neighbor_mask[P->pe] = 0; // never self

  std::vector<int> neighbors;
  for (int p = 0; p < P->npes; ++p)
    if (neighbor_mask[p])
      neighbors.push_back(p);
  P->n_neighbors = (int)neighbors.size();
  if (P->n_neighbors == 0) {
    // trivial case
    P->neighbors = P->recv_counts = P->recv_offsets = P->send_counts =
        P->send_offsets = nullptr;
    P->h_recv_ghost_ids = P->h_send_local_block_ids = nullptr;
    P->d_recv_ghost_ids = P->d_send_local_block_ids = nullptr;
    P->d_sendbuf = P->d_recvbuf = nullptr;
    return 0;
  }

  P->neighbors = (int *)std::malloc(P->n_neighbors * sizeof(int));
  P->recv_counts = (int *)std::calloc(P->n_neighbors, sizeof(int));
  P->send_counts = (int *)std::calloc(P->n_neighbors, sizeof(int));
  P->recv_offsets = (int *)std::malloc(P->n_neighbors * sizeof(int));
  P->send_offsets = (int *)std::malloc(P->n_neighbors * sizeof(int));

  // neighbor -> slot index map
  std::vector<int> slot(P->npes, -1);
  for (int a = 0; a < P->n_neighbors; ++a) {
    P->neighbors[a] = neighbors[a];
    slot[neighbors[a]] = a;
  }

  // Fill counts (recv: what we asked from owners; send: what others asked from
  // us)
  for (int p = 0; p < P->npes; ++p) {
    if (send_counts_req[p] > 0)
      P->recv_counts[slot[p]] = send_counts_req[p];
    if (recv_counts_req[p] > 0)
      P->send_counts[slot[p]] = recv_counts_req[p];
  }

  // Offsets + totals
  int total_recv = 0, total_send = 0;
  for (int a = 0; a < P->n_neighbors; ++a) {
    P->recv_offsets[a] = total_recv;
    total_recv += P->recv_counts[a];
  }
  for (int a = 0; a < P->n_neighbors; ++a) {
    P->send_offsets[a] = total_send;
    total_send += P->send_counts[a];
  }

  // ---- Build flattened recv ghost-ids (order EXACTLY matches the order we
  // requested) ----
  P->h_recv_ghost_ids = (int *)std::malloc(total_recv * sizeof(int));
  for (int p = 0; p < P->npes; ++p) {
    if (send_counts_req[p] == 0)
      continue;
    int a = slot[p];
    int off = P->recv_offsets[a];
    // our requests to p were contiguous starting at sdispl[p]
    std::memcpy(P->h_recv_ghost_ids + off, send_ghostids.data() + sdispl[p],
                send_counts_req[p] * sizeof(int));
  }

  // ---- Build flattened send local-block-ids by mapping received global ids
  // ----
  P->h_send_local_block_ids = (int *)std::malloc(total_send * sizeof(int));
  for (int p = 0; p < P->npes; ++p) {
    if (recv_counts_req[p] == 0)
      continue; // nobody requested from p
    int a = slot[p];
    int off = P->send_offsets[a];
    int cnt = recv_counts_req[p];
    int roff = rdispl[p];
    for (int i = 0; i < cnt; ++i) {
      gblk G = recv_gblks[roff + i];
      // sanity: we should own it
      int owner = owner_of_block(G, P->h_col_starts, P->npes);
      if (owner != P->pe) {
        fprintf(stderr, "[PE %d] ERROR: received request for non-owned block\n",
                P->pe);
        return -1;
      }
      int local_id = (int)(G - P->h_col_starts[P->pe]); // contiguous partition
      P->h_send_local_block_ids[off + i] = local_id;
    }
  }

  // ---- Upload device mirrors & scratch buffers ----
  CHECK_CUDA(
      cudaMalloc((void **)&P->d_recv_ghost_ids, total_recv * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void **)&P->d_send_local_block_ids,
                        total_send * sizeof(int)));
  if (total_recv > 0)
    CHECK_CUDA(cudaMemcpy(P->d_recv_ghost_ids, P->h_recv_ghost_ids,
                          total_recv * sizeof(int), cudaMemcpyHostToDevice));
  if (total_send > 0)
    CHECK_CUDA(cudaMemcpy(P->d_send_local_block_ids, P->h_send_local_block_ids,
                          total_send * sizeof(int), cudaMemcpyHostToDevice));

  CHECK_CUDA(
      cudaMalloc((void **)&P->d_sendbuf, total_send * P->bdim * sizeof(float)));
  CHECK_CUDA(
      cudaMalloc((void **)&P->d_recvbuf, total_recv * P->bdim * sizeof(float)));

  return 0;
}

int parbsr_halo_x(ParBSR *P) {
  CHECK_CUDA(cudaMemsetAsync(P->d_x_ghost, 0,
                             (size_t)P->nb_ghost * P->bdim * sizeof(float),
                             P->stream_comm));

  // Pack local blocks → sendbuf
  int total_send = 0, total_recv = 0;
  for (int a = 0; a < P->n_neighbors; ++a) {
    total_send += P->send_counts[a];
    total_recv += P->recv_counts[a];
  }

  int threads = 128;
  int s_off = 0;
  for (int a = 0; a < P->n_neighbors; ++a) {
    int cnt = P->send_counts[a];
    int n = cnt * P->bdim;
    if (n > 0) {
      pack_blocks_f32<<<(n + threads - 1) / threads, threads, 0,
                        P->stream_comm>>>(
          P->d_x_local, P->d_sendbuf + s_off * P->bdim,
          P->d_send_local_block_ids + P->send_offsets[a], cnt, P->bdim);
    }
    s_off += cnt;
  }

  // NCCL exchange (grouped)
  CHECK_NCCL(ncclGroupStart());
  int r_off = 0;
  for (int a = 0; a < P->n_neighbors; ++a) {
    int peer = P->neighbors[a];
    int cntb = P->recv_counts[a] * P->bdim;
    if (cntb > 0)
      CHECK_NCCL(ncclRecv(P->d_recvbuf + r_off, cntb, ncclFloat, peer, P->nccl,
                          P->stream_comm));
    r_off += cntb;
  }
  s_off = 0;
  for (int a = 0; a < P->n_neighbors; ++a) {
    int peer = P->neighbors[a];
    int cntb = P->send_counts[a] * P->bdim;
    if (cntb > 0)
      CHECK_NCCL(ncclSend(P->d_sendbuf + s_off, cntb, ncclFloat, peer, P->nccl,
                          P->stream_comm));
    s_off += cntb;
  }
  CHECK_NCCL(ncclGroupEnd());

  // Unpack recvbuf → x_ghost
  r_off = 0;
  for (int a = 0; a < P->n_neighbors; ++a) {
    int cnt = P->recv_counts[a];
    int n = cnt * P->bdim;
    if (n > 0) {
      unpack_blocks_f32<<<(n + threads - 1) / threads, threads, 0,
                          P->stream_comm>>>(
          P->d_recvbuf + r_off, P->d_x_ghost,
          P->d_recv_ghost_ids + P->recv_offsets[a], cnt, P->bdim);
    }
    r_off += n;
  }

  CHECK_CUDA(cudaStreamSynchronize(P->stream_comm));
  return 0;
}

int parbsr_spmv(ParBSR *P) {
  const float alpha = 1.0f, beta0 = 0.0f, beta1 = 1.0f;

  // y = A_ii * x_local
  CHECK_CUSPARSE(cusparseSbsrmv(
      P->sp_handle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
      P->mb_local, P->nb_local, P->Aii_nnzb, &alpha,
      (cusparseMatDescr_t)P->descrAii, P->d_Aii_val, P->d_Aii_rowptr,
      P->d_Aii_colind, P->bdim, P->d_x_local, &beta0, P->d_y));

  // y += A_ij * x_ghost
  if (P->Aij_nnzb > 0) {
    CHECK_CUSPARSE(cusparseSbsrmv(
        P->sp_handle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
        P->mb_local, P->nb_ghost, P->Aij_nnzb, &alpha,
        (cusparseMatDescr_t)P->descrAij, P->d_Aij_val, P->d_Aij_rowptr,
        P->d_Aij_colind, P->bdim, P->d_x_ghost, &beta1, P->d_y));
  }

  CHECK_CUDA(cudaStreamSynchronize(P->stream_comp));
  return 0;
}

void parbsr_print_vec(const ParBSR *P, const char *name, const float *dptr,
                      int n) {
  if (n <= 0) {
    printf("[PE %d] %s: (empty)\n", P->pe, name);
    return;
  }
  float *h = (float *)std::malloc(n * sizeof(float));
  cudaMemcpy(h, dptr, n * sizeof(float), cudaMemcpyDeviceToHost);
  printf("[PE %d] %s:", P->pe, name);
  for (int i = 0; i < n; ++i)
    printf(" %.1f", h[i]);
  printf("\n");
  std::free(h);
}

void parbsr_free(ParBSR *P) {
  if (!P)
    return;
  // device
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
  if (P->d_recv_ghost_ids)
    cudaFree(P->d_recv_ghost_ids);
  if (P->d_send_local_block_ids)
    cudaFree(P->d_send_local_block_ids);

  // host
  if (P->h_col_map_offd)
    free(P->h_col_map_offd);
  if (P->neighbors)
    free(P->neighbors);
  if (P->recv_counts)
    free(P->recv_counts);
  if (P->recv_offsets)
    free(P->recv_offsets);
  if (P->send_counts)
    free(P->send_counts);
  if (P->send_offsets)
    free(P->send_offsets);
  if (P->h_recv_ghost_ids)
    free(P->h_recv_ghost_ids);
  if (P->h_send_local_block_ids)
    free(P->h_send_local_block_ids);

  // handles
  if (P->descrAii)
    cusparseDestroyMatDescr((cusparseMatDescr_t)P->descrAii);
  if (P->descrAij)
    cusparseDestroyMatDescr((cusparseMatDescr_t)P->descrAij);
  if (P->sp_handle)
    cusparseDestroy(P->sp_handle);
  if (P->stream_comm)
    cudaStreamDestroy(P->stream_comm);
  if (P->stream_comp)
    cudaStreamDestroy(P->stream_comp);

  memset(P, 0, sizeof(*P));
}
