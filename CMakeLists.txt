cmake_minimum_required(VERSION 3.18)
project(par_bell_mpi LANGUAGES CXX CUDA)

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

# Optional MFEM support
set(MFEM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mfem-4.8" CACHE PATH "MFEM installation directory")
set(HYPRE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/hypre-2.26.0/src/hypre" CACHE PATH "Hypre installation directory")
set(METIS_DIR "$ENV{HOME}/opt/src/metis-5.1.0" CACHE PATH "METIS installation directory")

# NCCL: use NCCL_HOME or provide -DNCCL_ROOT=/path when configuring
set(NCCL_ROOT "$ENV{NCCL_HOME}" CACHE PATH "NCCL installation prefix")
if(NOT EXISTS "${NCCL_ROOT}/include/nccl.h")
  message(FATAL_ERROR "NCCL not found. Set NCCL_HOME env or pass -DNCCL_ROOT=/path")
endif()

# spdlog (header-only)
set(SPDLOG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/spdlog" CACHE PATH "spdlog directory")
if(NOT EXISTS "${SPDLOG_DIR}/include/spdlog/spdlog.h")
  message(FATAL_ERROR "spdlog not found at ${SPDLOG_DIR}. Clone into deps/spdlog")
endif()

add_library(parbell STATIC
  src/par_bell.cu
  src/par_csr.cu
  src/comm_pkg.cpp
  src/logging.cpp
)

target_include_directories(parbell
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${NCCL_ROOT}/include
    ${SPDLOG_DIR}/include
)

target_link_libraries(parbell
  PUBLIC
    CUDA::cudart
    CUDA::cusparse
    ${NCCL_ROOT}/lib/libnccl.so
    MPI::MPI_CXX
)

# RPATH so we don't need LD_LIBRARY_PATH for libnccl.so at runtime
target_link_options(parbell PRIVATE -Wl,-rpath,${NCCL_ROOT}/lib)

# Examples
add_executable(simple_example
  examples/simple.cpp
)

target_include_directories(simple_example
  PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${NCCL_ROOT}/include
    ${SPDLOG_DIR}/include
)

target_link_libraries(simple_example
  PRIVATE
    parbell
    MPI::MPI_CXX
)

# Ensure runtime can find libnccl.so via rpath
target_link_options(simple_example PRIVATE -Wl,-rpath,${NCCL_ROOT}/lib)

set_target_properties(simple_example parbell PROPERTIES
  CXX_STANDARD 17
  CUDA_SEPARABLE_COMPILATION ON
)

# MFEM-based example (always build; user-provided MFEM/Hypre expected)
add_executable(dg_diffusion
  examples/dg_diffusion.cpp
)

target_include_directories(dg_diffusion
  PRIVATE
    ${MFEM_DIR}
    ${MFEM_DIR}/include
    ${HYPRE_DIR}/include
)

target_link_libraries(dg_diffusion
  PRIVATE
    MPI::MPI_CXX
    ${MFEM_DIR}/libmfem.a
    ${HYPRE_DIR}/lib/libHYPRE.a
    ${METIS_DIR}/lib/libmetis.a
    OpenMP::OpenMP_CXX
    # Additional math libraries that MFEM/Hypre might need
    m
)

set_target_properties(dg_diffusion PROPERTIES
  CXX_STANDARD 17
)

# Tests: Poisson H1 CSR SpMV
add_executable(test_poisson
  tests/poisson.cpp
)

target_include_directories(test_poisson
  PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${MFEM_DIR}
    ${MFEM_DIR}/include
    ${HYPRE_DIR}/include
    ${NCCL_ROOT}/include
    ${SPDLOG_DIR}/include
)

target_link_libraries(test_poisson
  PRIVATE
    parbell
    MPI::MPI_CXX
    ${MFEM_DIR}/libmfem.a
    ${HYPRE_DIR}/lib/libHYPRE.a
    ${METIS_DIR}/lib/libmetis.a
    OpenMP::OpenMP_CXX
    m
)

set_target_properties(test_poisson PROPERTIES
  CXX_STANDARD 17
)

target_link_options(test_poisson PRIVATE -Wl,-rpath,${NCCL_ROOT}/lib)

# Benchmarks
add_executable(bench_poisson
  benchmarks/poisson_bench.cpp
)

target_include_directories(bench_poisson
  PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${MFEM_DIR}
    ${MFEM_DIR}/include
    ${HYPRE_DIR}/include
    ${NCCL_ROOT}/include
    ${SPDLOG_DIR}/include
)

target_link_libraries(bench_poisson
  PRIVATE
    parbell
    MPI::MPI_CXX
    ${MFEM_DIR}/libmfem.a
    ${HYPRE_DIR}/lib/libHYPRE.a
    ${METIS_DIR}/lib/libmetis.a
    OpenMP::OpenMP_CXX
    m
)

set_target_properties(bench_poisson PROPERTIES
  CXX_STANDARD 17
)

target_link_options(bench_poisson PRIVATE -Wl,-rpath,${NCCL_ROOT}/lib)
