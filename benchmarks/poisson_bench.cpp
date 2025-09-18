#include "mfem.hpp"
#include "par_csr.h"
#include "utils.h"

#include <vector>
#include <iostream>
#include <string>

using namespace mfem;
using namespace std;

int main(int argc, char* argv[]) {
  // Init MPI/Hypre
  Mpi::Init(argc, argv);
  Hypre::Init();
  int rank = Mpi::WorldRank();
  int size = Mpi::WorldSize();

  if (rank==0) logging::debug_logf(rank, "bench_poisson starting | ranks=%d", size);

  // Build mesh: 2 serial + 2 parallel refinements
  Mesh serial_mesh = Mesh::MakeCartesian3D(5, 5, 5, Element::TETRAHEDRON);
  for (int i = 0; i < 2; ++i) serial_mesh.UniformRefinement();
  ParMesh pmesh(MPI_COMM_WORLD, serial_mesh);
  serial_mesh.Clear();
  for (int i = 0; i < 3; ++i) pmesh.UniformRefinement();
  if (rank==0) logging::debug_logf(rank, "mesh built: dim=%d, elements=%d, vertices=%d", pmesh.Dimension(), pmesh.GetNE(), pmesh.GetNV());

  // H1 Poisson assembly with homogeneous Dirichlet
  H1_FECollection fec(1, pmesh.Dimension());
  ParFiniteElementSpace fes(&pmesh, &fec);
  Array<int> ess_tdof_list; fes.GetBoundaryTrueDofs(ess_tdof_list);
  if (rank==0) logging::debug_logf(rank, "FES true dofs=%lld", (long long)fes.GlobalTrueVSize());

  ParGridFunction x(&fes); x = 0.0;
  ParLinearForm b(&fes); ConstantCoefficient one(1.0); b.AddDomainIntegrator(new DomainLFIntegrator(one)); b.Assemble();
  ParBilinearForm a(&fes); a.AddDomainIntegrator(new DiffusionIntegrator); a.Assemble();

  HypreParMatrix A; Vector B, X_dummy;
  a.FormLinearSystem(ess_tdof_list, x, b, A, X_dummy, B);
  if (rank==0) logging::debug_logf(rank, "FormLinearSystem complete: B size=%lld", (long long)B.Size());

  // Build a non-pathological input vector X (random) to avoid near-nullspace issues
  Vector X(B.Size());
  X.Randomize(12345);
  // Reference SpMV: Y_ref = A * X
  Vector Y_ref(B.Size());

  // Initialize GPU device + NCCL (optional)
  init_gpu_for_rank();
  int use_mpi_halo = env_flag("HALO_USE_MPI", 0);
  ncclComm_t nccl = nullptr;
  if (!use_mpi_halo) {
    ncclUniqueId id; if (rank == 0) NCCLCHECK(ncclGetUniqueId(&id));
    CHECK_MPI(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    NCCLCHECK(ncclCommInitRank(&nccl, size, id, rank));
  }

  // Extract local matrices and col_starts
  SparseMatrix diag, offd; A.GetDiag(diag); HYPRE_BigInt* offd_cmap = nullptr; A.GetOffd(offd, offd_cmap);
  int m_local = diag.Height();
  int n_local = diag.Width();
  int nnzAii = diag.NumNonZeroElems();
  int nnzAij = offd.NumNonZeroElems();
  int n_ghost = offd.Width();
  long long local_nnz_ll = (long long)nnzAii + (long long)nnzAij;
  HYPRE_BigInt my_col_start = A.GetColStarts()[0];
  HYPRE_BigInt my_col_end   = A.GetColStarts()[1];
  int my_n_local = (int)(my_col_end - my_col_start);
  std::vector<int> all_n_local(size);
  CHECK_MPI(MPI_Allgather(&my_n_local, 1, MPI_INT, all_n_local.data(), 1, MPI_INT, MPI_COMM_WORLD));
  // Use the true global base offset from rank 0 to align with col_map_offd global ids
  HYPRE_BigInt base0 = 0;
  if (rank == 0) base0 = my_col_start;
  CHECK_MPI(MPI_Bcast(&base0, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD));
  std::vector<gblk> col_starts(size + 1, 0);
  col_starts[0] = (gblk)base0;
  for (int p = 0; p < size; ++p) col_starts[p + 1] = col_starts[p] + (gblk)all_n_local[p];
  long long global_nnz_est=0; CHECK_MPI(MPI_Allreduce(&local_nnz_ll,&global_nnz_est,1,MPI_LONG_LONG,MPI_SUM,MPI_COMM_WORLD));
  logging::debug_logf(rank, "matrix local: M=%d N=%d nnz(ii)=%d nnz(ij)=%d ghosts=%d | global nnz~=%lld", m_local, n_local, nnzAii, nnzAij, n_ghost, global_nnz_est);

  // Build ParCSR and upload mats
  ParCSR P; if (parcsr_init(&P, rank, size, m_local, n_local, col_starts.data(), nccl)) { MPI_Abort(MPI_COMM_WORLD, 2); }
  // Upload Aii
  std::vector<float> aii_vals(nnzAii); { const real_t* dv = diag.GetData(); for (int i = 0; i < nnzAii; ++i) aii_vals[i] = (float)dv[i]; }
  if (upload_csr(m_local, nnzAii, diag.GetI(), diag.GetJ(), aii_vals.data(), &P.d_Aii_rowptr, &P.d_Aii_colind, &P.d_Aii_val)) { MPI_Abort(MPI_COMM_WORLD, 3); }
  // Upload Aij
  std::vector<float> aij_vals(nnzAij);
  if (n_ghost > 0) {
    const real_t* dv = offd.GetData(); for (int i = 0; i < nnzAij; ++i) aij_vals[i] = (float)dv[i];
    if (upload_csr(m_local, nnzAij, offd.GetI(), offd.GetJ(), aij_vals.data(), &P.d_Aij_rowptr, &P.d_Aij_colind, &P.d_Aij_val)) { MPI_Abort(MPI_COMM_WORLD, 3); }
    P.n_ghost_host = n_ghost; P.n_ghost = n_ghost; P.h_col_map_offd = (gblk*)std::malloc((size_t)n_ghost * sizeof(gblk));
    for (int i = 0; i < n_ghost; ++i) P.h_col_map_offd[i] = (gblk)offd_cmap[i];
  }

  // Allocate vectors and copy X to device as input
  CUDACHECK(cudaMalloc((void**)&P.d_x_local, (size_t)n_local * sizeof(float)));
  CUDACHECK(cudaMalloc((void**)&P.d_x_ghost, (size_t)(P.n_ghost > 0 ? P.n_ghost : 1) * sizeof(float)));
  CUDACHECK(cudaMalloc((void**)&P.d_y,       (size_t)m_local * sizeof(float)));
  std::vector<float> x_local(n_local); for (int i = 0; i < n_local; ++i) x_local[i] = (float)X[i];
  CUDACHECK(cudaMemcpy(P.d_x_local, x_local.data(), (size_t)n_local * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(P.d_x_ghost, 0, (size_t)(P.n_ghost > 0 ? P.n_ghost : 1) * sizeof(float)));
  CUDACHECK(cudaMemset(P.d_y,       0, (size_t)m_local * sizeof(float)));

  if (parcsr_create_cusparse_descriptors(&P, nnzAii, nnzAij)) { MPI_Abort(MPI_COMM_WORLD, 4); }
  if (parcsr_build_comm_plan_from_colmap(&P, MPI_COMM_WORLD)) { logging::debug_logf(rank, "build comm plan failed"); MPI_Abort(MPI_COMM_WORLD, 5); }
  // Log plan summary
  {
    int ts = 0, tr = 0;
    for (int a = 0; a < P.plan.n_neighbors; ++a) { ts += P.plan.send_counts[a]; tr += P.plan.recv_counts[a]; }
    std::string s = "plan neighbors=" + std::to_string(P.plan.n_neighbors) + " totals send=" + std::to_string(ts) + " recv=" + std::to_string(tr) + " peers:";
    for (int a = 0; a < P.plan.n_neighbors; ++a) {
      s += " [p=" + std::to_string(P.plan.neighbors[a]) + ": s=" + std::to_string(P.plan.send_counts[a]) + ", r=" + std::to_string(P.plan.recv_counts[a]) + "]";
    }
    logging::debug_logf(rank, "%s", s.c_str());
  }
  // Pairwise validation now occurs inside comm plan build.

  // Timers and iteration count
  StopWatch t_hypre, t_gpu;
  const int iters = 3000;

  // Time Hypre SpMV: perform iters applications of A*X into Y_ref
  if (rank==0) logging::debug_logf(rank, "timing Hypre reference: iters=%d", iters);
  MPI_Barrier(MPI_COMM_WORLD);
  t_hypre.Clear(); t_hypre.Start();
  for (int it = 0; it < iters; ++it) {
    A.Mult(X, Y_ref);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  double hypre_sec = t_hypre.RealTime();
  // Global nnz for GFLOP/s estimation (2 flops per nonzero)
  long long local_nnz = (long long)nnzAii + (long long)nnzAij;
  long long global_nnz = 0;
  CHECK_MPI(MPI_Allreduce(&local_nnz, &global_nnz, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD));

  // Time our halo + SpMV on GPU for iters iterations
  if (rank==0) logging::debug_logf(rank, "timing custom GPU: iters=%d", iters);
  MPI_Barrier(MPI_COMM_WORLD);
  t_gpu.Clear(); t_gpu.Start();
  for (int it = 0; it < iters; ++it) {
    if (parcsr_halo_x(&P) != 0) { logging::debug_logf(rank, "parcsr_halo_x failed at iter %d", it); MPI_Abort(MPI_COMM_WORLD, 11); }
    if (parcsr_spmv(&P)   != 0) { logging::debug_logf(rank, "parcsr_spmv failed at iter %d", it);   MPI_Abort(MPI_COMM_WORLD, 12); }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  double ours_sec = t_gpu.RealTime();

  // Gather result and compute global L2 errors and norms
  std::vector<float> y(m_local, 0.0f);
  CUDACHECK(cudaMemcpy(y.data(), P.d_y, (size_t)m_local * sizeof(float), cudaMemcpyDeviceToHost));
  double local_err2 = 0.0, local_ref2 = 0.0, local_y2 = 0.0;
  for (int i = 0; i < m_local; ++i) {
    double di = (double)y[i] - (double)Y_ref[i];
    local_err2 += di * di;
    local_ref2 += (double)Y_ref[i] * (double)Y_ref[i];
    local_y2   += (double)y[i] * (double)y[i];
  }
  double global_err2 = 0.0, global_ref2 = 0.0, global_y2 = 0.0;
  CHECK_MPI(MPI_Allreduce(&local_err2, &global_err2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
  CHECK_MPI(MPI_Allreduce(&local_ref2, &global_ref2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
  CHECK_MPI(MPI_Allreduce(&local_y2,   &global_y2,   1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
  double l2 = std::sqrt(global_err2);
  double yref_l2 = std::sqrt(global_ref2);
  double y_l2    = std::sqrt(global_y2);
  double rel_l2  = (yref_l2 > 0.0) ? (l2 / yref_l2) : 0.0;

  if (rank == 0) {
    double flops_total = 2.0 * (double)global_nnz * (double)iters;
    double hypre_gflops = (hypre_sec > 0.0) ? (flops_total / hypre_sec / 1e9) : 0.0;
    double gpu_gflops   = (ours_sec  > 0.0) ? (flops_total / ours_sec  / 1e9) : 0.0;

    // Hypre CPU SpMV block (match hypre_gpu bench format)
    logging::debug_logf(rank, "Hypre CPU SpMV benchmark");
    logging::debug_logf(rank, "  MPI ranks: %d", size);
    logging::debug_logf(rank, "  DoFs:      %lld", (long long)fes.GlobalTrueVSize());
    logging::debug_logf(rank, "  iters:     %d", iters);
    logging::debug_logf(rank, "  total:     %.6f s", hypre_sec);
    logging::debug_logf(rank, "  per-iter:  %.6f us", 1e6 * hypre_sec / iters);
    logging::debug_logf(rank, "  GFLOP/s:   %.6f", hypre_gflops);

    // Custom GPU SpMV block (our implementation)
    logging::debug_logf(rank, "Custom GPU SpMV benchmark");
    logging::debug_logf(rank, "  MPI ranks: %d", size);
    logging::debug_logf(rank, "  DoFs:      %lld", (long long)fes.GlobalTrueVSize());
    logging::debug_logf(rank, "  iters:     %d", iters);
    logging::debug_logf(rank, "  total:     %.6f s", ours_sec);
    logging::debug_logf(rank, "  per-iter:  %.6f us", 1e6 * ours_sec / iters);
    logging::debug_logf(rank, "  GFLOP/s:   %.6f", gpu_gflops);

    // Accuracy summary
    logging::debug_logf(rank, "  ||y||=%.6e  ||y_ref||=%.6e  L2 err=%.6e  rel L2=%.6e", y_l2, yref_l2, l2, rel_l2);
  }

  parcsr_free(&P);
  MPI_Barrier(MPI_COMM_WORLD);
  if (nccl) ncclCommDestroy(nccl);
  return 0;
}
