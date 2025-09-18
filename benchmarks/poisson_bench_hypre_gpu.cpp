#include "mfem.hpp"
#include "utils.h"
#ifdef MPIX_CUDA_AWARE_SUPPORT
#include <mpi-ext.h>
#endif
#include <HYPRE_config.h>
#include <HYPRE_utilities.h>
#include <vector>
#include <iostream>

using namespace mfem;
using namespace std;

int main(int argc, char* argv[])
{
  // Initialize MPI and Hypre
  Mpi::Init(argc, argv);
  Hypre::Init();
  int rank = Mpi::WorldRank();
  int size = Mpi::WorldSize();

  // Enable MFEM CUDA backend (assumes MFEM and Hypre were built with CUDA)
  Device device("cuda");
  if (rank == 0) {
    logging::debug_logf(rank, "MFEM device backend: cuda");
    logging::debug_logf(rank, "sizeof(real_t)=%zu", sizeof(mfem::real_t));
  }

  // Optional toggle: select Hypre SpMV backend
  if (const char* env = std::getenv("HYPRE_SPMV_VENDOR"))
  {
    if (*env == '0' || *env == '1')
    {
      HYPRE_SetSpMVUseVendor((*env == '1') ? 1 : 0);
      if (rank == 0)
      {
        logging::debug_logf(rank, "HYPRE_SetSpMVUseVendor(%s)", ((*env=='1')?"1":"0"));
      }
    }
  }

  // Build mesh: start coarse, then refine (same spirit as original)
  Mesh serial_mesh = Mesh::MakeCartesian3D(5, 5, 5, Element::TETRAHEDRON);
  for (int i = 0; i < 2; ++i) { serial_mesh.UniformRefinement(); }
  ParMesh pmesh(MPI_COMM_WORLD, serial_mesh);
  serial_mesh.Clear();
  for (int i = 0; i < 3; ++i) { pmesh.UniformRefinement(); }

  // H1 Poisson assembly with homogeneous Dirichlet BCs
  H1_FECollection fec(1, pmesh.Dimension());
  ParFiniteElementSpace fes(&pmesh, &fec);
  Array<int> ess_tdof_list; fes.GetBoundaryTrueDofs(ess_tdof_list);

  ParGridFunction x0(&fes); x0 = 0.0;
  ParLinearForm b(&fes); ConstantCoefficient one(1.0);
  b.AddDomainIntegrator(new DomainLFIntegrator(one));
  b.Assemble();
  ParBilinearForm a(&fes); a.AddDomainIntegrator(new DiffusionIntegrator);
  a.Assemble();

  HypreParMatrix A; Vector B, X_dummy;
  a.FormLinearSystem(ess_tdof_list, x0, b, A, X_dummy, B);

  // Build a random input vector X to avoid near-nullspace pathologies
  Vector X(B.Size());
  Vector Y(B.Size());
  // Prefer device residency for vectors during timing
  X.UseDevice(true);
  Y.UseDevice(true);
  // Fill on host, then move to device and mark device copy current
  X.Randomize(12345);
  (void)X.Write(true);
  (void)Y.Write(true);

  // Prepare HypreParMatrix data for the memory class Hypre expects
  A.HypreRead();

  // Diagnostics: report device/memory classes and shallow-copy compatibility
  auto mc_to_str = [](mfem::MemoryClass mc) {
    switch (mc)
    {
      case mfem::MemoryClass::HOST:    return "HOST";
      case mfem::MemoryClass::DEVICE:  return "DEVICE";
      case mfem::MemoryClass::MANAGED: return "MANAGED";
    }
    return "UNKNOWN";
  };
  const bool dev_enabled = Device::IsEnabled();
  const mfem::MemoryClass hypre_mc = GetHypreMemoryClass();
  const bool x_host = IsHostMemory(X.GetMemory().GetMemoryType());
  const bool y_host = IsHostMemory(Y.GetMemory().GetMemoryType());
  const bool x_compatible = MemoryClassContainsType(hypre_mc,
                              X.GetMemory().GetMemoryType());
  const bool y_compatible = MemoryClassContainsType(hypre_mc,
                              Y.GetMemory().GetMemoryType());
  if (rank == 0)
  {
    logging::debug_logf(rank, "Device enabled: %s", (dev_enabled ? "yes" : "no"));
    logging::debug_logf(rank, "Hypre memory class: %s", mc_to_str(hypre_mc));
    logging::debug_logf(rank, "X host-mem-type: %s, class-compatible: %s", (x_host?"yes":"no"), (x_compatible?"yes":"no"));
    logging::debug_logf(rank, "Y host-mem-type: %s, class-compatible: %s", (y_host?"yes":"no"), (y_compatible?"yes":"no"));

    // Report MPI library version
    char mpi_ver[MPI_MAX_LIBRARY_VERSION_STRING]; int len = 0;
    MPI_Get_library_version(mpi_ver, &len);
    logging::debug_logf(rank, "MPI library: %.*s", len, mpi_ver);

    // Compile-time: was hypre built with GPU-aware MPI?
#ifdef HYPRE_USING_GPU_AWARE_MPI
    logging::debug_logf(rank, "HYPRE_USING_GPU_AWARE_MPI: yes");
#else
    logging::debug_logf(rank, "HYPRE_USING_GPU_AWARE_MPI: no");
#endif

    // Runtime (Open MPI): query CUDA-aware support if available
#ifdef MPIX_CUDA_AWARE_SUPPORT
    int cuda_aware = 0; MPIX_Query_cuda_support(&cuda_aware);
    logging::debug_logf(rank, "MPIX_Query_cuda_support: %s", (cuda_aware?"yes":"no"));
#endif
  }

  // Estimate global nnz for GF/s: nnz(diag) + nnz(offd)
  SparseMatrix diag, offd; A.GetDiag(diag);
  HYPRE_BigInt* offd_cmap = nullptr; A.GetOffd(offd, offd_cmap);
  long long local_nnz = (long long)diag.NumNonZeroElems() + (long long)offd.NumNonZeroElems();
  long long global_nnz = 0;
  MPI_Allreduce(&local_nnz, &global_nnz, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

  // Optional warmup to initialize GPU data structures
  for (int w = 0; w < 10; ++w) { A.Mult(X, Y); }

  // Re-check residency/compatibility after warmup in case anything migrated
  {
    const bool x_host2 = IsHostMemory(X.GetMemory().GetMemoryType());
    const bool y_host2 = IsHostMemory(Y.GetMemory().GetMemoryType());
    const bool x_compat2 = MemoryClassContainsType(hypre_mc,
                                X.GetMemory().GetMemoryType());
    const bool y_compat2 = MemoryClassContainsType(hypre_mc,
                                Y.GetMemory().GetMemoryType());
    if (rank == 0)
    {
      cout << "Post-warmup: X host-mem-type: " << (x_host2 ? "yes" : "no")
           << ", class-compatible: " << (x_compat2 ? "yes" : "no") << "\n"
           << "Post-warmup: Y host-mem-type: " << (y_host2 ? "yes" : "no")
           << ", class-compatible: " << (y_compat2 ? "yes" : "no") << "\n";
    }
  }

  // Time Hypre GPU SpMV: perform iters applications of A*X into Y
  const int iters = 3000;
  StopWatch t;
  MPI_Barrier(MPI_COMM_WORLD);
  t.Start();
  for (int it = 0; it < iters; ++it) { A.Mult(X, Y); }
  MPI_Barrier(MPI_COMM_WORLD);
  t.Stop();
  double sec = t.RealTime();

  if (rank == 0)
  {
    double flops_total = 2.0 * (double)global_nnz * (double)iters;
    double gflops = (sec > 0.0) ? (flops_total / sec / 1e9) : 0.0;
    logging::debug_logf(rank, "Hypre GPU SpMV benchmark");
    logging::debug_logf(rank, "  MPI ranks: %d", size);
    logging::debug_logf(rank, "  DoFs:      %lld", (long long)fes.GlobalTrueVSize());
    logging::debug_logf(rank, "  iters:     %d", iters);
    logging::debug_logf(rank, "  total:     %.6f s", sec);
    logging::debug_logf(rank, "  per-iter:  %.6f us", 1e6 * sec / iters);
    logging::debug_logf(rank, "  GFLOP/s:   %.6f", gflops);
  }

  return 0;
}
