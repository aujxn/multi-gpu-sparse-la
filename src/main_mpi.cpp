#include "par_bsr.h"

#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <cstdio>
#include <vector>
#include <cstdlib>

#define CHECK_MPI(x) do{ int e=(x); if(e!=MPI_SUCCESS){ \
  fprintf(stderr,"MPI error %s:%d (code %d)\n",__FILE__,__LINE__,e); MPI_Abort(MPI_COMM_WORLD,e);} }while(0)
#define CHECK_CUDA(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD,1);} }while(0)
#define CHECK_NCCL(x) do{ ncclResult_t n=(x); if(n!=ncclSuccess){ \
  fprintf(stderr,"NCCL %s:%d: %s\n",__FILE__,__LINE__,ncclGetErrorString(n)); MPI_Abort(MPI_COMM_WORLD,1);} }while(0)

int main(int argc, char** argv)
{
  CHECK_MPI(MPI_Init(&argc, &argv));

  int world_rank=0, world_size=1;
  CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
  CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

  // Split to get local rank per node (portable)
  MPI_Comm local_comm;
  CHECK_MPI(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank,
                                MPI_INFO_NULL, &local_comm));
  int local_rank=0;
  CHECK_MPI(MPI_Comm_rank(local_comm, &local_rank));
  CHECK_MPI(MPI_Comm_free(&local_comm));

  int dev_count=0;
  CHECK_CUDA(cudaGetDeviceCount(&dev_count));
  if (dev_count<=0) {
    if (world_rank==0) fprintf(stderr,"No GPUs visible to process.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  int dev = local_rank % dev_count;
  CHECK_CUDA(cudaSetDevice(dev));

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

  ParBSR P;
  if (parbsr_init(&P, world_rank, world_size, mb_local, bdim, col_starts.data(), comm)) {
    fprintf(stderr,"[rank %d] parbsr_init failed\n", world_rank);
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if (parbsr_build_toy(&P)) {
    fprintf(stderr,"[rank %d] parbsr_build_toy failed\n", world_rank);
    MPI_Abort(MPI_COMM_WORLD, 3);
  }
  //if (parbsr_build_comm_plan_ring(&P)) {
  //fprintf(stderr,"[rank %d] parbsr_build_comm_plan_ring failed\n", world_rank);
  //MPI_Abort(MPI_COMM_WORLD, 4);
  //}
  if (parbsr_build_comm_plan_from_colmap(&P, MPI_COMM_WORLD)) {
    fprintf(stderr,"[rank %d] parbsr_build_comm_plan_from_colmap failed\n", world_rank);
    MPI_Abort(MPI_COMM_WORLD, 4);
  }


  MPI_Barrier(MPI_COMM_WORLD);
  if (P.pe==0) printf("---- ring plan ----\n");
  printf("[PE %d] neighbors:", P.pe);
  for (int a=0;a<P.n_neighbors;++a)
    printf(" %d recv=%d send=%d", P.neighbors[a], P.recv_counts[a], P.send_counts[a]);
  printf("\n");
  MPI_Barrier(MPI_COMM_WORLD);

  // Halo then SpMV
  if (parbsr_halo_x(&P)) {
    fprintf(stderr,"[rank %d] halo failed\n", world_rank);
    MPI_Abort(MPI_COMM_WORLD, 5);
  }
  if (parbsr_spmv(&P)) {
    fprintf(stderr,"[rank %d] spmv failed\n", world_rank);
    MPI_Abort(MPI_COMM_WORLD, 6);
  }

  // Debug print
  MPI_Barrier(MPI_COMM_WORLD);
  int n_local = P.mb_local * P.bdim;
  int n_ghost = P.nb_ghost  * P.bdim;
  parbsr_print_vec(&P, "x_local", P.d_x_local, n_local);
  if (n_ghost>0) parbsr_print_vec(&P, "x_ghost", P.d_x_ghost, n_ghost);
  parbsr_print_vec(&P, "y",       P.d_y,       n_local);
  MPI_Barrier(MPI_COMM_WORLD);

  parbsr_free(&P);
  ncclCommDestroy(comm);
  MPI_Finalize();
  return 0;
}

