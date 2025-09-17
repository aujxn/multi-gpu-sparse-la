// MFEM/Hypre
#include "mfem.hpp"

// Our GPU CSR path
#include "par_csr.h"
#include "utils.h"

#include <fstream>
#include <iostream>
#include <vector>

using namespace std;
using namespace mfem;

static void run_parcsr_spmv_test(HypreParMatrix &A, const Vector &X_in, const Vector &Y_ref);

int main(int argc, char *argv[]) {
  // Init MPI/Hypre
  Mpi::Init(argc, argv);
  Hypre::Init();

  // Problem setup: simple H1 Poisson with Dirichlet BCs
  const int order = 1;
  Mesh serial_mesh = Mesh::MakeCartesian3D(5, 5, 5, Element::TETRAHEDRON);
  ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
  serial_mesh.Clear();

  H1_FECollection fec(order, mesh.Dimension());
  ParFiniteElementSpace fespace(&mesh, &fec);

  Array<int> ess_tdof_list; // all boundary
  fespace.GetBoundaryTrueDofs(ess_tdof_list);

  ParGridFunction x(&fespace); x = 0.0;

  ConstantCoefficient one(1.0);
  ParLinearForm b(&fespace); b.AddDomainIntegrator(new DomainLFIntegrator(one)); b.Assemble();

  ParBilinearForm a(&fespace); a.AddDomainIntegrator(new DiffusionIntegrator); a.Assemble();

  HypreParMatrix A;
  Vector B, X;
  a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

  if (Mpi::Root()) {
    cout << "Size of linear system: " << A.Height() << endl;
  }

  // Reference SpMV via Hypre: use RHS as input to avoid zero vector
  // Y_ref = A * B
  Vector Y_ref(B.Size());
  A.Mult(B, Y_ref);

  // Run GPU CSR test and compare Y using input B
  run_parcsr_spmv_test(A, B, Y_ref);
  return 0;
}

static void run_parcsr_spmv_test(HypreParMatrix &A, const Vector &X_in, const Vector &Y_ref) {
  // 1) Initialize GPU + NCCL (1 GPU per local rank)
  init_gpu_for_rank();
  int world_rank = Mpi::WorldRank();
  int world_size = Mpi::WorldSize();

  ncclUniqueId id;
  if (world_rank == 0) NCCLCHECK(ncclGetUniqueId(&id));
  CHECK_MPI(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
  ncclComm_t nccl;
  NCCLCHECK(ncclCommInitRank(&nccl, world_size, id, world_rank));

  // 2) Extract Hypre diag/offd parts and column map
  SparseMatrix diag, offd;
  A.GetDiag(diag);
  HYPRE_BigInt *offd_cmap = nullptr;
  A.GetOffd(offd, offd_cmap);

  const int m_local = diag.Height();
  const int n_local = diag.Width();
  const int nnzAii  = diag.NumNonZeroElems();
  const int nnzAij  = offd.NumNonZeroElems();
  const int n_ghost = offd.Width();

  // Row/col starts: build global col_starts across ranks
  const HYPRE_BigInt my_col_start = A.GetColStarts()[0];
  const HYPRE_BigInt my_col_end   = A.GetColStarts()[1];
  const int          my_n_local   = (int)(my_col_end - my_col_start);

  std::vector<int> all_n_local(world_size);
  CHECK_MPI(MPI_Allgather(&my_n_local, 1, MPI_INT, all_n_local.data(), 1, MPI_INT, MPI_COMM_WORLD));
  HYPRE_BigInt start0 = 0;
  if (world_rank == 0) start0 = my_col_start;
  CHECK_MPI(MPI_Bcast(&start0, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD));
  std::vector<gblk> col_starts(world_size + 1, 0);
  col_starts[0] = (gblk)start0;
  for (int p = 0; p < world_size; ++p) col_starts[p + 1] = col_starts[p] + (gblk)all_n_local[p];

  // 3) Create ParCSR and upload matrices
  ParCSR P;
  if (parcsr_init(&P, world_rank, world_size, m_local, n_local, col_starts.data(), nccl)) {
    fprintf(stderr, "[rank %d] parcsr_init failed\n", world_rank);
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  // Upload A_ii CSR (must cast double -> float)
  std::vector<float> aii_vals(nnzAii);
  {
    const real_t *dv = diag.GetData();
    for (int i = 0; i < nnzAii; ++i) aii_vals[i] = (float)dv[i];
    if (upload_csr(m_local, nnzAii, diag.GetI(), diag.GetJ(), aii_vals.data(),
                   &P.d_Aii_rowptr, &P.d_Aii_colind, &P.d_Aii_val)) {
      fprintf(stderr, "[rank %d] upload_csr Aii failed\n", world_rank);
      MPI_Abort(MPI_COMM_WORLD, 3);
    }
  }

  // Upload A_ij CSR and ghost map (if present)
  std::vector<float> aij_vals(nnzAij);
  if (n_ghost > 0) {
    const real_t *dv = offd.GetData();
    for (int i = 0; i < nnzAij; ++i) aij_vals[i] = (float)dv[i];
    if (upload_csr(m_local, nnzAij, offd.GetI(), offd.GetJ(), aij_vals.data(),
                   &P.d_Aij_rowptr, &P.d_Aij_colind, &P.d_Aij_val)) {
      fprintf(stderr, "[rank %d] upload_csr Aij failed\n", world_rank);
      MPI_Abort(MPI_COMM_WORLD, 3);
    }
    P.n_ghost_host = n_ghost;
    P.n_ghost      = n_ghost;
    P.h_col_map_offd = (gblk*)std::malloc((size_t)n_ghost * sizeof(gblk));
    for (int i = 0; i < n_ghost; ++i) P.h_col_map_offd[i] = (gblk)offd_cmap[i];
  } else {
    P.n_ghost_host = 0; P.n_ghost = 0; P.h_col_map_offd = nullptr;
  }

  // 4) Allocate vectors and copy input X_in (here, the RHS B)
  CUDACHECK(cudaMalloc((void**)&P.d_x_local, (size_t)n_local * sizeof(float)));
  CUDACHECK(cudaMalloc((void**)&P.d_x_ghost, (size_t)(P.n_ghost > 0 ? P.n_ghost : 1) * sizeof(float)));
  CUDACHECK(cudaMalloc((void**)&P.d_y,       (size_t)m_local * sizeof(float)));
  std::vector<float> x_local(n_local);
  for (int i = 0; i < n_local; ++i) x_local[i] = (float)X_in[i];
  CUDACHECK(cudaMemcpy(P.d_x_local, x_local.data(), (size_t)n_local * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMemset(P.d_x_ghost, 0, (size_t)(P.n_ghost > 0 ? P.n_ghost : 1) * sizeof(float)));
  CUDACHECK(cudaMemset(P.d_y,       0, (size_t)m_local * sizeof(float)));

  // 5) Create cuSPARSE descriptors and comm plan, halo exchange, SpMV
  if (parcsr_create_cusparse_descriptors(&P, nnzAii, nnzAij)) {
    fprintf(stderr, "[rank %d] create descriptors failed\n", world_rank);
    MPI_Abort(MPI_COMM_WORLD, 4);
  }
  if (parcsr_build_comm_plan_from_colmap(&P, MPI_COMM_WORLD)) {
    fprintf(stderr, "[rank %d] build comm plan failed\n", world_rank);
    MPI_Abort(MPI_COMM_WORLD, 5);
  }
  // Cross-rank pairwise count validation
  if (comm_plan_validate_pairwise(&P.plan, world_rank, P.npes, MPI_COMM_WORLD) != 0) {
    debug_logf(world_rank, "comm_plan_validate_pairwise failed");
    MPI_Abort(MPI_COMM_WORLD, 5);
  }
  // Extra validation: ghost global IDs should be within [col_starts[0], col_starts[npes])
  if (P.n_ghost_host > 0 && P.h_col_map_offd) {
    gblk gmin = P.h_col_map_offd[0], gmax = P.h_col_map_offd[0];
    for (int i = 1; i < P.n_ghost_host; ++i) { gblk g = P.h_col_map_offd[i]; if (g < gmin) gmin = g; if (g > gmax) gmax = g; }
    if (!(gmin >= P.h_col_starts[0] && gmax < P.h_col_starts[P.npes])) {
      debug_logf(world_rank, "ghost global IDs out of range: min=%lld max=%lld range=[%lld,%lld)",
                 (long long)gmin, (long long)gmax,
                 (long long)P.h_col_starts[0], (long long)P.h_col_starts[P.npes]);
      MPI_Abort(MPI_COMM_WORLD, 5);
    }
  }
  if (parcsr_halo_x(&P)) {
    fprintf(stderr, "[rank %d] halo_x failed\n", world_rank);
    MPI_Abort(MPI_COMM_WORLD, 6);
  }
  if (parcsr_spmv(&P)) {
    fprintf(stderr, "[rank %d] spmv failed\n", world_rank);
    MPI_Abort(MPI_COMM_WORLD, 7);
  }

  // 6) Compare with reference
  std::vector<float> y(m_local, 0.0f);
  CUDACHECK(cudaMemcpy(y.data(), P.d_y, (size_t)m_local * sizeof(float), cudaMemcpyDeviceToHost));
  double max_abs_diff = 0.0;
  for (int i = 0; i < m_local; ++i) {
    double yi = (double)y[i];
    double yr = (double)Y_ref[i];
    double di = std::abs(yi - yr);
    max_abs_diff = std::max(max_abs_diff, di);
    // Per-element explicit log for this rank
    debug_logf(world_rank, "compare i=%d: y=%.9e ref=%.9e diff=%.3e", i, yi, yr, di);
  }
  // Build MFEM vectors for norm computations (convert float->real_t)
  std::vector<real_t> y_host_rt(m_local);
  for (int i = 0; i < m_local; ++i) y_host_rt[i] = (real_t)y[i];
  Vector y_vec(y_host_rt.data(), m_local);
  Vector err(y_vec);
  err -= Y_ref;
  double err_l2 = err.Norml2();
  double yref_l2 = Y_ref.Norml2();
  double rel_l2 = (yref_l2 > 0.0) ? (err_l2 / yref_l2) : 0.0;
  if (Mpi::Root()) {
    mfem::out << "CSR SpMV verification max|diff| = " << max_abs_diff
              << ", L2 error = " << err_l2
              << ", rel L2 = " << rel_l2 << std::endl;
  }
  // Also log per-rank for easier collection
  debug_logf(world_rank, "CSR SpMV: max|diff|=%.6e L2=%.6e relL2=%.6e", max_abs_diff, err_l2, rel_l2);

  // 7) Cleanup
  parcsr_free(&P);
  ncclCommDestroy(nccl);
}
