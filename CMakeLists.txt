cmake_minimum_required(VERSION 3.18)
project(par_bsr_mpi LANGUAGES CXX CUDA)

find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)

# NCCL: use NCCL_HOME or provide -DNCCL_ROOT=/path when configuring
set(NCCL_ROOT "$ENV{NCCL_HOME}" CACHE PATH "NCCL installation prefix")
if(NOT EXISTS "${NCCL_ROOT}/include/nccl.h")
  message(FATAL_ERROR "NCCL not found. Set NCCL_HOME env or pass -DNCCL_ROOT=/path")
endif()

add_library(parbsr STATIC
  src/par_bsr.cu
)

target_include_directories(parbsr
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${NCCL_ROOT}/include
)

target_link_libraries(parbsr
  PUBLIC
    CUDA::cudart
    CUDA::cusparse
    ${NCCL_ROOT}/lib/libnccl.so
    MPI::MPI_CXX
)

# RPATH so we don't need LD_LIBRARY_PATH for libnccl.so at runtime
target_link_options(parbsr PRIVATE -Wl,-rpath,${NCCL_ROOT}/lib)

add_executable(par_bsr_mpi
  src/main_mpi.cpp
)

target_include_directories(par_bsr_mpi
  PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${NCCL_ROOT}/include
)

target_link_libraries(par_bsr_mpi
  PRIVATE
    parbsr
    MPI::MPI_CXX
)

set_target_properties(par_bsr_mpi parbsr PROPERTIES
  CXX_STANDARD 17
  CUDA_SEPARABLE_COMPILATION ON
)
