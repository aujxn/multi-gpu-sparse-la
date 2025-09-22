#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <nccl.h>
#include <mpi.h>
#include <cstdlib>
#include <stdarg.h>

// CUDA error checking
#define CHECK_CUDA(x)                                                          \
  do {                                                                         \
    cudaError_t e = (x);                                                       \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,                 \
              cudaGetErrorString(e));                                          \
      return -1;                                                               \
    }                                                                          \
  } while (0)

// cuSPARSE error checking
#define CHECK_CUSPARSE(x)                                                      \
  do {                                                                         \
    cusparseStatus_t s = (x);                                                  \
    if (s != CUSPARSE_STATUS_SUCCESS) {                                        \
      fprintf(stderr, "cuSPARSE %s:%d: %d\n", __FILE__, __LINE__, (int)s);    \
      return -1;                                                               \
    }                                                                          \
  } while (0)

// NCCL error checking
#define CHECK_NCCL(x)                                                          \
  do {                                                                         \
    ncclResult_t n = (x);                                                      \
    if (n != ncclSuccess) {                                                    \
      fprintf(stderr, "NCCL %s:%d: %s\n", __FILE__, __LINE__,                 \
              ncclGetErrorString(n));                                          \
      return -1;                                                               \
    }                                                                          \
  } while (0)

// MPI error checking
#define CHECK_MPI(x)                                                           \
  do {                                                                         \
    int e = (x);                                                               \
    if (e != MPI_SUCCESS) {                                                    \
      fprintf(stderr, "MPI error %s:%d (code %d)\n", __FILE__, __LINE__, e);  \
      MPI_Abort(MPI_COMM_WORLD, e);                                            \
    }                                                                          \
  } while (0)

// Alternative NCCL check that exits instead of returning (for main functions)
#define NCCLCHECK(cmd)                                                         \
  do {                                                                         \
    ncclResult_t r = cmd;                                                      \
    if (r != ncclSuccess) {                                                    \
      printf("Failed, NCCL error %s:%d '%s'\n",                               \
             __FILE__, __LINE__, ncclGetErrorString(r));                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Alternative CUDA check that exits instead of returning (for main functions)
#define CUDACHECK(cmd)                                                         \
  do {                                                                         \
    cudaError_t e = cmd;                                                       \
    if (e != cudaSuccess) {                                                    \
      printf("Failed, CUDA error %s:%d: %s\n",                                \
             __FILE__, __LINE__, cudaGetErrorString(e));                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/**
 * Initialize GPU device for current MPI rank.
 * This function:
 * 1. Gets local rank within each node
 * 2. Validates enough GPUs are available for unique assignment
 * 3. Sets the CUDA device for this rank based on local rank
 * 
 * Each local rank gets a unique GPU. Errors if there aren't enough GPUs
 * on the node for all local ranks to have unique assignments.
 */
static inline void init_gpu_for_rank() {
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get local rank per node (portable across different MPI implementations)
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank,
                        MPI_INFO_NULL, &local_comm);
    int local_rank, local_size;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_size);
    MPI_Comm_free(&local_comm);

    // Check GPU availability
    int dev_count;
    CUDACHECK(cudaGetDeviceCount(&dev_count));
    
    if (dev_count <= 0) {
        fprintf(stderr, "[RANK %d] ERROR: No GPUs visible to process.\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Ensure enough GPUs for unique assignment to each local rank
    if (local_size > dev_count) {
        fprintf(stderr, "[RANK %d] ERROR: %d local ranks on node but only %d GPUs available.\n"
                        "Need at least %d GPUs per node for unique GPU assignment.\n",
                world_rank, local_size, dev_count, local_size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Assign unique GPU to this local rank
    int dev = local_rank % dev_count;
    CUDACHECK(cudaSetDevice(dev));

    if (world_rank == 0) {
        printf("GPU initialization: %d ranks across nodes, %d GPUs per node, unique assignment\n", 
               world_size, dev_count);
    }
}

/**
 * Get the local rank within the current node.
 * Useful for debugging and logging purposes.
 */
static inline int get_local_rank() {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank,
                        MPI_INFO_NULL, &local_comm);
    int local_rank;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);

    return local_rank;
}

static inline int env_flag(const char* name, int defv=0) {
    const char* v = std::getenv(name);
    if (!v) return defv;
    return (int)strtol(v, nullptr, 10);
}

// Append a formatted debug line to logs with rank prefix
void debug_logf(int pe, const char* fmt, ...);
namespace logging { void debug_logf(int pe, const char* fmt, ...); }

// Lightweight stage logger that reports timing deltas and optional memory info
#include <chrono>
class StageLogger {
public:
    explicit StageLogger(int rank, bool debug_mem)
      : rank_(rank), debug_mem_(debug_mem), start_(clock::now()), last_(start_) {}

    // Log a stage with time since previous log and optional memory diagnostics
    void log(const char* stage);

private:
    using clock = std::chrono::steady_clock;
    int rank_;
    bool debug_mem_;
    clock::time_point start_;
    clock::time_point last_;
};
