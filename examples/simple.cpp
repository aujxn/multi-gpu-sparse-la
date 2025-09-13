/*
Dense toy layout per rank (mb_local=2, bdim=2, nb_local=2, nb_ghost∈{0,1})


8 ranks:

2   |                                       000
 2  |                                       001
  2 |  11                                   002
   2|  11                                   003
----------------------------------------
    |2   |                                  100
    | 2  |                                  101
    |  2 |  11                              102
    |   2|  11                              103
----------------------------------------
         |2   |                             200
         | 2  |                             201
         |  2 |  11                         202
         |   2|  11                         203
----------------------------------------
              |2   |                        300
              | 2  |                        301
              |  2 |  11                    302
              |   2|  11                    303
----------------------------------------
                   |2   |                   400
                   | 2  |                   401
                   |  2 |  11               402
                   |   2|  11               403
----------------------------------------
                        |2   |              500
                        | 2  |              501
                        |  2 |  11          502
                        |   2|  11          503
----------------------------------------
                             |2   |         600
                             | 2  |         601
                             |  2 |  11     602
                             |   2|  11     603
----------------------------------------
                                  |2   |    700
                                  | 2  |    701
                                  |  2 |    702
                                  |   2|    703
*/
#include "par_bell.h"
#include "utils.h"

#include <cstdio>
#include <vector>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <string>
#include <algorithm>

int main(int argc, char** argv)
{
  CHECK_MPI(MPI_Init(&argc, &argv));

  // Initialize GPU device for this rank (enforces 1 GPU per process)
  init_gpu_for_rank();

  int world_rank=0, world_size=1;
  CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
  CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

  // Bootstrap NCCL across all ranks
  ncclUniqueId id;
  if (world_rank == 0) CHECK_NCCL(ncclGetUniqueId(&id));
  CHECK_MPI(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  ncclComm_t comm;
  CHECK_NCCL(ncclCommInitRank(&comm, world_size, id, world_rank));

  // Toy global block partition: each rank owns 2 block-cols (contiguous)
  const int bdim      = 2;    // block size
  const int mb_local  = 2;    // two block rows per rank
  const int blocks_per_rank = 2;
  std::vector<gblk> col_starts(world_size+1);
  for (int r=0; r<=world_size; ++r) col_starts[r] = (gblk)(r*blocks_per_rank);

  ParBELL P;
  if (parbell_init(&P, world_rank, world_size, mb_local, bdim, col_starts.data(), comm)) {
    fprintf(stderr,"[rank %d] parbsr_init failed\n", world_rank);
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  // Build toy Blocked-ELL in example: Aii = 2*I (ell_cols=1),
  // Aij = ones block at last row to ghost block 0 (if right neighbor exists)
  {
    P.Aii_ell_cols = 1;
    int mb = P.mb_local;
    std::vector<int> h_ci_ii(mb * P.Aii_ell_cols, 0);
    std::vector<float> h_val_ii(mb * P.Aii_ell_cols * P.bdim * P.bdim, 0.0f);
    for (int br = 0; br < mb; ++br) {
      h_ci_ii[0 * mb + br] = br; // diagonal block column
      float *blk = &h_val_ii[(0 * mb + br) * (P.bdim * P.bdim)];
      for (int r = 0; r < P.bdim; ++r)
        for (int c = 0; c < P.bdim; ++c)
          blk[r * P.bdim + c] = (r == c) ? 2.0f : 0.0f;
    }
    if (upload_blocked_ell(P.mb_local, P.Aii_ell_cols, P.bdim,
                           h_ci_ii.data(), h_val_ii.data(),
                           &P.d_Aii_colind, &P.d_Aii_val)) {
      fprintf(stderr,"[rank %d] upload Aii failed\n", world_rank);
      MPI_Abort(MPI_COMM_WORLD, 3);
    }

    int has_right = (P.pe + 1 < P.npes) ? 1 : 0;
    if (has_right) {
      P.Aij_ell_cols = 1;
      P.nb_ghost_host = 1;
      P.nb_ghost = 1;
      std::vector<int> h_ci_ij(mb * P.Aij_ell_cols, 0);
      std::vector<float> h_val_ij(mb * P.Aij_ell_cols * P.bdim * P.bdim, 0.0f);
      int last = mb - 1;
      h_ci_ij[0 * mb + last] = 0; // ghost block 0
      float *blk = &h_val_ij[(0 * mb + last) * (P.bdim * P.bdim)];
      for (int i = 0; i < P.bdim * P.bdim; ++i) blk[i] = 1.0f;
      if (upload_blocked_ell(P.mb_local, P.Aij_ell_cols, P.bdim,
                             h_ci_ij.data(), h_val_ij.data(),
                             &P.d_Aij_colind, &P.d_Aij_val)) {
        fprintf(stderr,"[rank %d] upload Aij failed\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 3);
      }
      P.h_col_map_offd = (gblk*)std::malloc(sizeof(gblk));
      P.h_col_map_offd[0] = P.h_col_starts[P.pe + 1];
    } else {
      P.Aij_ell_cols = 0;
      P.nb_ghost_host = 0;
      P.nb_ghost = 0;
      P.d_Aij_colind = nullptr;
      P.d_Aij_val = nullptr;
      P.h_col_map_offd = nullptr;
    }

    // Allocate vectors and initialize
    int n_local = P.mb_local * P.bdim;
    int n_ghost = P.nb_ghost * P.bdim;
    CUDACHECK(cudaMalloc((void **)&P.d_x_local, n_local * sizeof(float)));
    CUDACHECK(cudaMalloc((void **)&P.d_x_ghost, (n_ghost > 0 ? n_ghost : 1) * sizeof(float)));
    CUDACHECK(cudaMalloc((void **)&P.d_y, n_local * sizeof(float)));
    std::vector<float> xh(n_local);
    for (int i = 0; i < n_local; ++i) xh[i] = 100.0f * P.pe + (float)i;
    CUDACHECK(cudaMemcpy(P.d_x_local, xh.data(), n_local * sizeof(float), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemset(P.d_x_ghost, 0, (n_ghost > 0 ? n_ghost : 1) * sizeof(float)));
    CUDACHECK(cudaMemset(P.d_y, 0, n_local * sizeof(float)));

    // Create cuSPARSE descriptors (SpMat + DnMat) and allocate workspace
    if (create_cusparse_descriptors(&P) != 0) {
      fprintf(stderr,"[rank %d] create_cusparse_descriptors failed\n", world_rank);
      MPI_Abort(MPI_COMM_WORLD, 3);
    }
    // Brief dims summary for clarity
    int M = P.mb_local * P.bdim;
    int Kloc = P.nb_local * P.bdim;
    int Kgh = P.nb_ghost * P.bdim;
    debug_logf(P.pe, "Descriptors ready: M=%d K_local=%d K_ghost=%d bdim=%d Aii_ell_cols=%d Aij_ell_cols=%d",
               M, Kloc, Kgh, P.bdim, P.Aii_ell_cols, P.Aij_ell_cols);
  }
  if (parbell_build_comm_plan_from_colmap(&P, MPI_COMM_WORLD)) {
    fprintf(stderr,"[rank %d] parbsr_build_comm_plan_from_colmap failed\n", world_rank);
    MPI_Abort(MPI_COMM_WORLD, 4);
  }


  MPI_Barrier(MPI_COMM_WORLD);
  if (P.pe==0) debug_logf(P.pe, "---- setup ----");
  // Dump partitioning and dims
  {
    std::string s = "col_starts:";
    char buf[64];
    for (int r=0;r<=world_size;++r) {
      int n = snprintf(buf, sizeof(buf), " %lld", (long long)col_starts[r]);
      s.append(buf, buf+n);
    }
    debug_logf(P.pe, "%s", s.c_str());
  }
  debug_logf(P.pe, "mb_local=%d nb_local=%d nb_ghost=%d bdim=%d Aii_ell_cols=%d Aij_ell_cols=%d",
             P.mb_local, P.nb_local, P.nb_ghost, P.bdim, P.Aii_ell_cols, P.Aij_ell_cols);
  if (P.nb_ghost>0 && P.h_col_map_offd) {
    std::string s = "col_map_offd (global IDs):";
    char buf[64];
    for (int i=0;i<P.nb_ghost;++i) {
      int n = snprintf(buf, sizeof(buf), " %lld", (long long)P.h_col_map_offd[i]);
      s.append(buf, buf+n);
    }
    debug_logf(P.pe, "%s", s.c_str());
  }

  debug_logf(P.pe, "---- neighbors and comm plan ----");
  {
    std::string s = "neighbors:";
    char buf[64];
    for (int a=0;a<P.plan.n_neighbors;++a) {
      int n = snprintf(buf, sizeof(buf), " %d recv=%d send=%d", P.plan.neighbors[a], P.plan.recv_counts[a], P.plan.send_counts[a]);
      s.append(buf, buf+n);
    }
    debug_logf(P.pe, "%s", s.c_str());
  }
  // Dump flattened lists
  {
    int total_recv = comm_plan_total_recv(&P.plan);
    int total_send = comm_plan_total_send(&P.plan);
    std::string r = "recv_ghost_ids:";
    char buf[64];
    for (int i=0;i<total_recv;++i) { int n=snprintf(buf,sizeof(buf)," %d", P.plan.h_recv_ids?P.plan.h_recv_ids[i]:-1); r.append(buf,buf+n);}    
    debug_logf(P.pe, "%s", r.c_str());
    std::string s = "send_local_block_ids:";
    for (int i=0;i<total_send;++i) { int n=snprintf(buf,sizeof(buf)," %d", P.plan.h_send_ids?P.plan.h_send_ids[i]:-1); s.append(buf,buf+n);}    
    debug_logf(P.pe, "%s", s.c_str());
  }

  // Dump BELL structure (colind arrays and select blocks)
  {
    // A_ii
    if (P.Aii_ell_cols>0) {
      std::vector<int> h_ci(P.mb_local*P.Aii_ell_cols);
      CUDACHECK(cudaMemcpy(h_ci.data(), P.d_Aii_colind, h_ci.size()*sizeof(int), cudaMemcpyDeviceToHost));
      for (int i=0;i<P.mb_local;++i) {
        std::string line = "Aii row "+std::to_string(i)+" colind:";
        char buf[64];
        for (int j=0;j<P.Aii_ell_cols;++j) { int n=snprintf(buf,sizeof(buf)," %d", h_ci[j*P.mb_local + i]); line.append(buf,buf+n);}        
        debug_logf(P.pe, "%s", line.c_str());
      }
    }
    // A_ij
    if (P.Aij_ell_cols>0 && P.nb_ghost>0) {
      std::vector<int> h_ci(P.mb_local*P.Aij_ell_cols);
      CUDACHECK(cudaMemcpy(h_ci.data(), P.d_Aij_colind, h_ci.size()*sizeof(int), cudaMemcpyDeviceToHost));
      for (int i=0;i<P.mb_local;++i) {
        std::string line = "Aij row "+std::to_string(i)+" colind:";
        char buf[64];
        for (int j=0;j<P.Aij_ell_cols;++j) { int n=snprintf(buf,sizeof(buf)," %d", h_ci[j*P.mb_local + i]); line.append(buf,buf+n);}        
        debug_logf(P.pe, "%s", line.c_str());
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Halo then SpMV
  if (parbell_halo_x(&P)) {
    fprintf(stderr,"[rank %d] halo failed\n", world_rank);
    MPI_Abort(MPI_COMM_WORLD, 5);
  }
  if (parbell_spmv(&P)) {
    fprintf(stderr,"[rank %d] spmv failed\n", world_rank);
    MPI_Abort(MPI_COMM_WORLD, 6);
  }

  // Debug print
  MPI_Barrier(MPI_COMM_WORLD);
  int n_local = P.mb_local * P.bdim;
  int n_ghost = P.nb_ghost  * P.bdim;
  parbell_print_vec(&P, "x_local", P.d_x_local, n_local);
  if (n_ghost>0) parbell_print_vec(&P, "x_ghost", P.d_x_ghost, n_ghost);
  parbell_print_vec(&P, "y",       P.d_y,       n_local);
  MPI_Barrier(MPI_COMM_WORLD);
  // Verify expected result for the toy matrix
  {
    std::vector<float> y_host(n_local);
    CUDACHECK(cudaMemcpy(y_host.data(), P.d_y, n_local*sizeof(float), cudaMemcpyDeviceToHost));

    // Expected: y = 2*x_local; plus, in last block-row, add Ones*ghost_block0 if has right neighbor
    std::vector<float> y_exp(n_local);
    for (int i=0;i<n_local;++i) y_exp[i] = 2.0f * (100.0f*P.pe + (float)i);
    if (P.nb_ghost > 0) {
      float sumg = 0.0f;
      for (int r=0;r<P.bdim;++r) sumg += 100.0f*(P.pe+1) + (float)r;
      int base = (P.mb_local-1)*P.bdim;
      for (int r=0;r<P.bdim;++r) y_exp[base + r] += sumg;
    }

    // Also print the expected vector for debugging
    {
      std::string s = "y_truth:";
      char buf[64];
      for (int i=0;i<n_local;++i) {
        int n = snprintf(buf, sizeof(buf), " %.1f", y_exp[i]);
        s.append(buf, buf+n);
      }
      debug_logf(P.pe, "%s", s.c_str());
    }

    double max_abs_diff = 0.0;
    for (int i=0;i<n_local;++i) {
      max_abs_diff = std::max(max_abs_diff, (double)std::abs(y_host[i]-y_exp[i]));
    }
    if (max_abs_diff > 1e-4) {
      fprintf(stderr, "[rank %d] verification FAILED: max|diff|=%.6g\n", world_rank, max_abs_diff);
    }
    debug_logf(P.pe, "verification max|diff| = %.6g", max_abs_diff);
  }

  parbell_free(&P);
  ncclCommDestroy(comm);
  MPI_Finalize();
  return 0;
}
