#include "mfem.hpp"
#include "utils.h"
#ifdef MPIX_CUDA_AWARE_SUPPORT
#include <mpi-ext.h>
#endif
#include <HYPRE_config.h>
#include <HYPRE_utilities.h>
#include <vector>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace mfem;
using namespace std;

int main(int argc, char* argv[])
{
  // Initialize MPI and Hypre
  Mpi::Init(argc, argv);
  Hypre::Init();
  int rank = Mpi::WorldRank();
  int size = Mpi::WorldSize();
  const int debug_mem = env_flag("DEBUG_MEM", 0);
  StageLogger stage(rank, debug_mem);
  // Report OpenMP threading
  int omp_threads = 1;
  int omp_max_threads = 1;
  int omp_num_procs = 1;
#ifdef _OPENMP
#pragma omp parallel
  {
#pragma omp master
    { omp_threads = omp_get_num_threads(); }
  }
  omp_max_threads = omp_get_max_threads();
  omp_num_procs = omp_get_num_procs();
#endif
  const char* omp_env = std::getenv("OMP_NUM_THREADS");
  const char* slurm_cpt = std::getenv("SLURM_CPUS_PER_TASK");
  if (rank==0) logging::debug_logf(rank, "OpenMP threads: team=%d max=%d procs=%d (OMP_NUM_THREADS=%s SLURM_CPUS_PER_TASK=%s)",
                                   omp_threads, omp_max_threads, omp_num_procs,
                                   omp_env?omp_env:"unset", slurm_cpt?slurm_cpt:"unset");

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

  // Build mesh: configurable base grid and refinements via env
  const int nx = env_flag("MESH_NX", 5);
  const int ny = env_flag("MESH_NY", nx);
  const int nz = env_flag("MESH_NZ", nx);
  const int sref = env_flag("SERIAL_REFS", 2);
  const int pref = env_flag("PAR_REFS", 3);
  if (rank == 0) { logging::debug_logf(rank, "mesh params: nx=%d ny=%d nz=%d sref=%d pref=%d", nx, ny, nz, sref, pref); }
  Mesh serial_mesh = Mesh::MakeCartesian3D(nx, ny, nz, Element::TETRAHEDRON);
  for (int i = 0; i < sref; ++i) { serial_mesh.UniformRefinement(); }
  ParMesh pmesh(MPI_COMM_WORLD, serial_mesh);
  serial_mesh.Clear();
  for (int i = 0; i < pref; ++i) { pmesh.UniformRefinement(); }
  if (rank == 0) { logging::debug_logf(rank, "mesh built: dim=%d, elements=%d, vertices=%d", pmesh.Dimension(), pmesh.GetNE(), pmesh.GetNV()); }
  stage.log("mesh_ready");

  // H1 Poisson assembly with homogeneous Dirichlet BCs
  H1_FECollection fec(1, pmesh.Dimension());
  ParFiniteElementSpace fes(&pmesh, &fec);
  Array<int> ess_tdof_list; fes.GetBoundaryTrueDofs(ess_tdof_list);
  stage.log("fes_ready");

  ParGridFunction x0(&fes); x0 = 0.0;
  ParLinearForm b(&fes); ConstantCoefficient one(1.0);
  b.AddDomainIntegrator(new DomainLFIntegrator(one));
  b.Assemble();
  ParBilinearForm a(&fes); a.AddDomainIntegrator(new DiffusionIntegrator);
  a.Assemble();
  stage.log("a_assembled");

  HypreParMatrix A; Vector B, X_dummy;
  a.FormLinearSystem(ess_tdof_list, x0, b, A, X_dummy, B);
  stage.log("linear_system");

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
  if (rank == 0) { logging::debug_logf(rank, "global nnz: %lld", global_nnz); }

  // Optional warmup to initialize GPU data structures
  for (int w = 0; w < 10; ++w) { A.Mult(X, Y); }
  stage.log("warmup_done");

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
  const int iters = env_flag("GPU_ITERS", 3000);
  StopWatch t;
  MPI_Barrier(MPI_COMM_WORLD);
  t.Start();
  for (int it = 0; it < iters; ++it) { A.Mult(X, Y); }
  MPI_Barrier(MPI_COMM_WORLD);
  t.Stop();
  double sec = t.RealTime();
  stage.log("spmv_done");

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
