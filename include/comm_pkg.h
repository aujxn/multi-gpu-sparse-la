#pragma once
#include <mpi.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef long long gblk; // global block/column id

// Reusable communication plan for halo exchanges of x
typedef struct {
  // Neighbor topology
  int   n_neighbors;
  int*  neighbors;      // peer ranks, length n_neighbors
  int*  recv_counts;    // items received from neighbor a
  int*  recv_offsets;   // displs into flattened recv arrays
  int*  send_counts;    // items sent to neighbor a
  int*  send_offsets;   // displs into flattened send arrays
  int   total_recv;     // cached sum(recv_counts)
  int   total_send;     // cached sum(send_counts)

  // Flattened index lists (host)
  int*  h_recv_ids;     // indices in x_ghost to fill (length=sum recv_counts)
  int*  h_send_ids;     // indices in x_local to gather (length=sum send_counts)

  // Device mirrors (indices only). Data buffers are owned by caller.
  int*  d_recv_ids;
  int*  d_send_ids;
} CommPlan;

// Build a plan from a list of required global column IDs (ghost map) and a
// contiguous ownership map col_starts over npes ranks. Fills both host and
// device index arrays.
int comm_plan_build_from_colmap(CommPlan* plan,
                                int pe, int npes,
                                int nb_ghost_host,
                                const gblk* h_col_map_offd,
                                const gblk* h_col_starts,
                                MPI_Comm comm);

// Query helpers
int comm_plan_total_recv(const CommPlan* plan);
int comm_plan_total_send(const CommPlan* plan);

// Free all resources in the plan
void comm_plan_free(CommPlan* plan);

// Cross-rank validation: ensure pairwise counts match between peers
int comm_plan_validate_pairwise(const CommPlan* plan, int pe, int npes, MPI_Comm comm);

#ifdef __cplusplus
} // extern "C"
#endif
