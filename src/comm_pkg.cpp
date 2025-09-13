#include "comm_pkg.h"
#include "utils.h"

#include <vector>
#include <cstring>

static inline int owner_of_block(gblk g, const gblk* col_starts, int npes) {
  for (int p = 0; p < npes; ++p)
    if (g >= col_starts[p] && g < col_starts[p + 1])
      return p;
  return -1;
}

int comm_plan_total_recv(const CommPlan* plan) {
  if (!plan || plan->n_neighbors == 0) return 0;
  return plan->total_recv;
}

int comm_plan_total_send(const CommPlan* plan) {
  if (!plan || plan->n_neighbors == 0) return 0;
  return plan->total_send;
}

int comm_plan_build_from_colmap(CommPlan* plan,
                                int pe, int npes,
                                int nb_ghost_host,
                                const gblk* h_col_map_offd,
                                const gblk* h_col_starts,
                                MPI_Comm comm) {
  if (!plan) return -1;
  std::memset(plan, 0, sizeof(*plan));

  // Group requested global ids by owner
  std::vector<int> recv_counts_vec(npes, 0);
  std::vector<std::vector<gblk>> req_gblks(npes);
  std::vector<std::vector<int>>  req_ghostids(npes);

  for (int g = 0; g < nb_ghost_host; ++g) {
    gblk G = h_col_map_offd[g];
    int owner = owner_of_block(G, h_col_starts, npes);
    if (owner == pe) continue; // sanity
    recv_counts_vec[owner] += 1;
    req_gblks[owner].push_back(G);
    req_ghostids[owner].push_back(g);
  }

  // Exchange counts (how many ids each peer requests from us)
  std::vector<int> send_counts_req(npes, 0);
  for (int p = 0; p < npes; ++p) send_counts_req[p] = recv_counts_vec[p];
  std::vector<int> recv_counts_req(npes, 0);
  CHECK_MPI(MPI_Alltoall(send_counts_req.data(), 1, MPI_INT,
                         recv_counts_req.data(), 1, MPI_INT, comm));

  // Compute displacements
  std::vector<int> sdispl(npes, 0), rdispl(npes, 0);
  int send_total = 0, recv_total = 0;
  for (int p = 1; p < npes; ++p) sdispl[p] = sdispl[p-1] + send_counts_req[p-1];
  for (int p = 1; p < npes; ++p) rdispl[p] = rdispl[p-1] + recv_counts_req[p-1];
  for (int p = 0; p < npes; ++p) { send_total += send_counts_req[p]; recv_total += recv_counts_req[p]; }

  // Flatten outgoing requests and ghost-id order
  std::vector<gblk> send_gblks(send_total);
  std::vector<int>  send_ghostids(send_total);
  for (int p = 0; p < npes; ++p) {
    int off = sdispl[p];
    for (size_t i = 0; i < req_gblks[p].size(); ++i) {
      send_gblks[off + (int)i] = req_gblks[p][i];
      send_ghostids[off + (int)i] = req_ghostids[p][i];
    }
  }

  // Exchange requested global ids
  std::vector<gblk> recv_gblks(recv_total);
  MPI_Datatype mpi_gblk = (sizeof(gblk) == 8) ? MPI_LONG_LONG : MPI_INT;
  CHECK_MPI(MPI_Alltoallv(send_gblks.data(), send_counts_req.data(), sdispl.data(), mpi_gblk,
                          recv_gblks.data(), recv_counts_req.data(), rdispl.data(), mpi_gblk,
                          comm));

  // Build neighbor list
  std::vector<char> neighbor_mask(npes, 0);
  for (int p = 0; p < npes; ++p) if (send_counts_req[p] > 0) neighbor_mask[p] = 1; // we recv from p
  for (int p = 0; p < npes; ++p) if (recv_counts_req[p] > 0) neighbor_mask[p] = 1; // we send to p
  neighbor_mask[pe] = 0;

  std::vector<int> neighbors;
  for (int p = 0; p < npes; ++p) if (neighbor_mask[p]) neighbors.push_back(p);

  plan->n_neighbors = (int)neighbors.size();
  if (plan->n_neighbors == 0) return 0; // nothing to do

  plan->neighbors    = (int*)std::malloc(plan->n_neighbors * sizeof(int));
  plan->recv_counts  = (int*)std::calloc(plan->n_neighbors, sizeof(int));
  plan->recv_offsets = (int*)std::malloc(plan->n_neighbors * sizeof(int));
  plan->send_counts  = (int*)std::calloc(plan->n_neighbors, sizeof(int));
  plan->send_offsets = (int*)std::malloc(plan->n_neighbors * sizeof(int));
  plan->total_recv   = 0;
  plan->total_send   = 0;

  std::vector<int> slot(npes, -1);
  for (int a = 0; a < plan->n_neighbors; ++a) { plan->neighbors[a] = neighbors[a]; slot[neighbors[a]] = a; }

  for (int p = 0; p < npes; ++p) {
    if (send_counts_req[p] > 0) plan->recv_counts[slot[p]] = send_counts_req[p];
    if (recv_counts_req[p] > 0) plan->send_counts[slot[p]] = recv_counts_req[p];
  }

  int total_recv = 0, total_send = 0;
  for (int a = 0; a < plan->n_neighbors; ++a) { plan->recv_offsets[a] = total_recv; total_recv += plan->recv_counts[a]; }
  for (int a = 0; a < plan->n_neighbors; ++a) { plan->send_offsets[a] = total_send; total_send += plan->send_counts[a]; }
  plan->total_recv = total_recv;
  plan->total_send = total_send;

  // Build flattened recv ghost-ids matching our request order to each owner
  plan->h_recv_ids = (int*)std::malloc((size_t)total_recv * sizeof(int));
  for (int p = 0; p < npes; ++p) {
    if (send_counts_req[p] == 0) continue;
    int a = slot[p];
    std::memcpy(plan->h_recv_ids + plan->recv_offsets[a],
                send_ghostids.data() + sdispl[p],
                (size_t)send_counts_req[p] * sizeof(int));
  }

  // Build flattened send local indices by mapping received global ids
  plan->h_send_ids = (int*)std::malloc((size_t)total_send * sizeof(int));
  for (int p = 0; p < npes; ++p) {
    if (recv_counts_req[p] == 0) continue;
    int a = slot[p];
    int off = plan->send_offsets[a];
    int cnt = recv_counts_req[p];
    int roff = rdispl[p];
    for (int i = 0; i < cnt; ++i) {
      gblk G = recv_gblks[roff + i];
      int owner = owner_of_block(G, h_col_starts, npes);
      (void)owner; // owner should be 'pe'
      int local_id = (int)(G - h_col_starts[pe]);
      plan->h_send_ids[off + i] = local_id;
    }
  }

  // Upload device mirrors (indices only)
  if (total_recv > 0) CHECK_CUDA(cudaMalloc((void**)&plan->d_recv_ids, (size_t)total_recv * sizeof(int)));
  if (total_send > 0) CHECK_CUDA(cudaMalloc((void**)&plan->d_send_ids, (size_t)total_send * sizeof(int)));
  if (total_recv > 0) CHECK_CUDA(cudaMemcpy(plan->d_recv_ids, plan->h_recv_ids, (size_t)total_recv * sizeof(int), cudaMemcpyHostToDevice));
  if (total_send > 0) CHECK_CUDA(cudaMemcpy(plan->d_send_ids, plan->h_send_ids, (size_t)total_send * sizeof(int), cudaMemcpyHostToDevice));

  // Final cross-rank validation now happens here (pairwise counts agreement)
  if (comm_plan_validate_pairwise(plan, pe, npes, comm) != 0) {
    return -1;
  }
  return 0;
}

void comm_plan_free(CommPlan* plan) {
  if (!plan) return;
  if (plan->neighbors)     free(plan->neighbors);
  if (plan->recv_counts)   free(plan->recv_counts);
  if (plan->recv_offsets)  free(plan->recv_offsets);
  if (plan->send_counts)   free(plan->send_counts);
  if (plan->send_offsets)  free(plan->send_offsets);
  if (plan->h_recv_ids)    free(plan->h_recv_ids);
  if (plan->h_send_ids)    free(plan->h_send_ids);
  if (plan->d_recv_ids)    cudaFree(plan->d_recv_ids);
  if (plan->d_send_ids)    cudaFree(plan->d_send_ids);
  std::memset(plan, 0, sizeof(*plan));
}

int comm_plan_validate_pairwise(const CommPlan* plan, int pe, int npes, MPI_Comm comm) {
  if (!plan) return -1;
  // Build full-size arrays of counts to/from every peer
  std::vector<int> send_to(npes, 0), recv_from(npes, 0);
  for (int a = 0; a < plan->n_neighbors; ++a) {
    int p = plan->neighbors[a];
    if (p < 0 || p >= npes) return -1;
    send_to[p]  = plan->send_counts[a];
    recv_from[p]= plan->recv_counts[a];
  }
  // Exchange our send_to via Alltoall; peers report what they expect to receive from us.
  std::vector<int> peers_expect_from_me(npes, 0);
  CHECK_MPI(MPI_Alltoall(send_to.data(), 1, MPI_INT, peers_expect_from_me.data(), 1, MPI_INT, comm));
  // Exchange our recv_from via Alltoall; peers tell us how many they intend to send us.
  std::vector<int> peers_send_to_me(npes, 0);
  CHECK_MPI(MPI_Alltoall(recv_from.data(), 1, MPI_INT, peers_send_to_me.data(), 1, MPI_INT, comm));

  // Validate both views match
  for (int p = 0; p < npes; ++p) {
    if (send_to[p] != peers_send_to_me[p]) {
      debug_logf(pe, "Comm validation mismatch to peer %d: our send=%d, their report send_to_me=%d", p, send_to[p], peers_send_to_me[p]);
      return -1;
    }
    if (recv_from[p] != peers_expect_from_me[p]) {
      debug_logf(pe, "Comm validation mismatch from peer %d: our recv=%d, their expect_from_me=%d", p, recv_from[p], peers_expect_from_me[p]);
      return -1;
    }
  }
  return 0;
}
