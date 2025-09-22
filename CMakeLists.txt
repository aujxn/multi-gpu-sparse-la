cmake_minimum_required(VERSION 3.18)
project(par_bell_mpi LANGUAGES CXX CUDA)

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

# Optional MFEM support
set(MFEM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mfem-4.8" CACHE PATH "MFEM installation directory")

# Prefer explicit env-provided HYPRE_DIR. Otherwise, try sensible defaults.
set(HYPRE_DIR_DEFAULT "$ENV{HYPRE_DIR}")
if(NOT HYPRE_DIR_DEFAULT)
  if(EXISTS "$ENV{HOME}/opt/src/hypre/include/HYPRE.h")
    set(HYPRE_DIR_DEFAULT "$ENV{HOME}/opt/src/hypre")
  elseif(EXISTS "$ENV{HOME}/opt/src/hypre/src/hypre/include/HYPRE.h")
    set(HYPRE_DIR_DEFAULT "$ENV{HOME}/opt/src/hypre/src/hypre")
  endif()
endif()
set(HYPRE_DIR "${HYPRE_DIR_DEFAULT}" CACHE PATH "Hypre installation directory")

if(NOT EXISTS "${HYPRE_DIR}/include/HYPRE.h")
  message(FATAL_ERROR "HYPRE not found. Set -DHYPRE_DIR=/path/to/hypre[/src/hypre] (with include/HYPRE.h and lib/libHYPRE.a)")
endif()

set(METIS_DIR "$ENV{HOME}/opt/src/metis-5.1.0" CACHE PATH "METIS installation directory")

# NCCL: locate headers and library. Prefer NCCL_HOME; otherwise search $HOME/opt/src
set(_NCCL_HOME "$ENV{NCCL_HOME}")
set(_NCCL_HINT "$ENV{HOME}/opt/src")
find_path(NCCL_INCLUDE_DIR nccl.h
  HINTS
    ${_NCCL_HOME}/include
    ${_NCCL_HOME}
    ${_NCCL_HINT}
    ${_NCCL_HINT}/nccl
  PATH_SUFFIXES include
)
find_library(NCCL_LIBRARY nccl
  HINTS
    ${_NCCL_HOME}/lib
    ${_NCCL_HOME}
    ${_NCCL_HINT}
    ${_NCCL_HINT}/nccl
  PATH_SUFFIXES lib lib64
)
if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
  message(FATAL_ERROR "NCCL not found. Set NCCL_HOME or install under $ENV{HOME}/opt/src (need include/nccl.h and lib/libnccl.so)")
endif()
get_filename_component(NCCL_LIB_DIR "${NCCL_LIBRARY}" DIRECTORY)

# spdlog (header-only) – default to $HOME/opt/src/spdlog if not provided
set(SPDLOG_DIR_DEFAULT "$ENV{SPDLOG_DIR}")
if(NOT SPDLOG_DIR_DEFAULT)
  set(SPDLOG_DIR_DEFAULT "$ENV{HOME}/opt/src/spdlog")
endif()
set(SPDLOG_DIR "${SPDLOG_DIR_DEFAULT}" CACHE PATH "spdlog directory")
if(NOT EXISTS "${SPDLOG_DIR}/include/spdlog/spdlog.h")
  message(FATAL_ERROR "spdlog not found at ${SPDLOG_DIR}. Set -DSPDLOG_DIR=/path/to/spdlog (header-only)")
endif()

add_library(parbell STATIC
  src/par_bell.cu
  src/par_csr.cu
  src/comm_pkg.cpp
  src/logging.cpp
  src/stage_logger.cpp
)

target_include_directories(parbell
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${NCCL_INCLUDE_DIR}
    ${SPDLOG_DIR}/include
)

target_link_libraries(parbell
  PUBLIC
    CUDA::cudart
    CUDA::cusparse
    ${NCCL_LIBRARY}
    MPI::MPI_CXX
)

# RPATH so we don't need LD_LIBRARY_PATH for libnccl.so at runtime
target_link_options(parbell PRIVATE -Wl,-rpath,${NCCL_LIB_DIR})

# Examples
add_executable(simple_example
  examples/simple.cpp
)

target_include_directories(simple_example
  PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${NCCL_INCLUDE_DIR}
    ${SPDLOG_DIR}/include
)

target_link_libraries(simple_example
  PRIVATE
    parbell
    MPI::MPI_CXX
)

# Ensure runtime can find libnccl.so via rpath
target_link_options(simple_example PRIVATE -Wl,-rpath,${NCCL_LIB_DIR})

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
    ${HYPRE_DIR}/include
    ${MFEM_DIR}
    ${MFEM_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(dg_diffusion
  PRIVATE
    MPI::MPI_CXX
    ${MFEM_DIR}/libmfem.a
    ${HYPRE_DIR}/lib/libHYPRE.a
    ${METIS_DIR}/lib/libmetis.a
    OpenMP::OpenMP_CXX
    CUDA::cudart
    CUDA::cusparse
    CUDA::cublas
    CUDA::curand
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
    ${HYPRE_DIR}/include
    ${MFEM_DIR}
    ${MFEM_DIR}/include
    ${NCCL_INCLUDE_DIR}
    ${SPDLOG_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(test_poisson
  PRIVATE
    parbell
    MPI::MPI_CXX
    ${MFEM_DIR}/libmfem.a
    ${HYPRE_DIR}/lib/libHYPRE.a
    ${METIS_DIR}/lib/libmetis.a
    OpenMP::OpenMP_CXX
    CUDA::cudart
    CUDA::cusparse
    CUDA::cublas
    CUDA::curand
    m
)

set_target_properties(test_poisson PROPERTIES
  CXX_STANDARD 17
)

target_link_options(test_poisson PRIVATE -Wl,-rpath,${NCCL_LIB_DIR})

# Benchmarks
add_executable(bench_poisson
  benchmarks/poisson_bench.cpp
)

target_include_directories(bench_poisson
  PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${HYPRE_DIR}/include
    ${MFEM_DIR}
    ${MFEM_DIR}/include
    ${NCCL_INCLUDE_DIR}
    ${SPDLOG_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(bench_poisson
  PRIVATE
    parbell
    MPI::MPI_CXX
    ${MFEM_DIR}/libmfem.a
    ${HYPRE_DIR}/lib/libHYPRE.a
    ${METIS_DIR}/lib/libmetis.a
    OpenMP::OpenMP_CXX
    CUDA::cudart
    CUDA::cusparse
    CUDA::cublas
    CUDA::curand
    m
)

set_target_properties(bench_poisson PROPERTIES
  CXX_STANDARD 17
)

target_link_options(bench_poisson PRIVATE -Wl,-rpath,${NCCL_LIB_DIR})

# Hypre GPU-only benchmark (MFEM/Hypre on device)
add_executable(bench_hypre_gpu
  benchmarks/poisson_bench_hypre_gpu.cpp
)

target_include_directories(bench_hypre_gpu
  PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${HYPRE_DIR}/include
    ${MFEM_DIR}
    ${MFEM_DIR}/include
    ${NCCL_INCLUDE_DIR}
    ${SPDLOG_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(bench_hypre_gpu
  PRIVATE
    parbell
    MPI::MPI_CXX
    ${MFEM_DIR}/libmfem.a
    ${HYPRE_DIR}/lib/libHYPRE.a
    ${METIS_DIR}/lib/libmetis.a
    OpenMP::OpenMP_CXX
    CUDA::cudart
    CUDA::cusparse
    CUDA::cublas
    CUDA::curand
    m
)

set_target_properties(bench_hypre_gpu PROPERTIES
  CXX_STANDARD 17
)

target_link_options(bench_hypre_gpu PRIVATE -Wl,-rpath,${NCCL_LIB_DIR})
