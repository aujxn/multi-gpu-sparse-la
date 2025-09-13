#include "par_bsr.h"
#include "utils.h"
#include "mfem.hpp"
#include <cmath>
#include <algorithm>
#include <vector>

// Hypre includes
#include "_hypre_parcsr_mv.h"

using namespace mfem;

int parbsr_from_hypre_parmatrix(ParBSR* P, 
                                int pe, int npes, 
                                HypreParMatrix* hypre_A,
                                ncclComm_t nccl_comm)
{
    if (!P || !hypre_A) return -1;
    
    // Get the underlying hypre_ParCSRMatrix
    hypre_ParCSRMatrix* par_A = (hypre_ParCSRMatrix*) (*hypre_A);
    hypre_CSRMatrix* diag = hypre_ParCSRMatrixDiag(par_A);
    hypre_CSRMatrix* offd = hypre_ParCSRMatrixOffd(par_A);
    
    // Matrix dimensions
    int local_rows = hypre_CSRMatrixNumRows(diag);
    int local_cols = hypre_CSRMatrixNumCols(diag);
    int offd_cols = hypre_CSRMatrixNumCols(offd);
    
    // For now, assume block size = 1 (CSR -> BSR with 1x1 blocks)
    int bdim = 1;
    
    // Setup global column starts (contiguous partitioning assumption)
    std::vector<gblk> col_starts(npes + 1);
    HYPRE_BigInt* row_starts = hypre_ParCSRMatrixRowStarts(par_A);
    HYPRE_BigInt* col_starts_big = hypre_ParCSRMatrixColStarts(par_A);
    
    for (int i = 0; i <= npes; i++) {
        col_starts[i] = (gblk)col_starts_big[i];
    }
    
    // Initialize ParBSR structure
    if (parbsr_init(P, pe, npes, local_rows, bdim, col_starts.data(), nccl_comm) != 0) {
        return -1;
    }
    
    // Extract diagonal part (A_ii)
    int* diag_i = hypre_CSRMatrixI(diag);
    int* diag_j = hypre_CSRMatrixJ(diag);
    double* diag_data = hypre_CSRMatrixData(diag);
    int diag_nnz = hypre_CSRMatrixNumNonzeros(diag);
    
    P->Aii_nnzb = diag_nnz;  // Each CSR entry becomes a 1x1 block
    
    // Convert CSR to BSR format (diagonal)
    std::vector<int> bsr_diag_rowptr(local_rows + 1);
    std::vector<int> bsr_diag_colind(diag_nnz);
    std::vector<float> bsr_diag_val(diag_nnz);  // 1x1 blocks
    
    for (int i = 0; i <= local_rows; i++) {
        bsr_diag_rowptr[i] = diag_i[i];
    }
    
    for (int k = 0; k < diag_nnz; k++) {
        bsr_diag_colind[k] = diag_j[k];
        bsr_diag_val[k] = (float)diag_data[k];
    }
    
    // Upload diagonal part to device
    if (upload_bsr(local_rows, P->Aii_nnzb, bdim, 
                   bsr_diag_rowptr.data(), bsr_diag_colind.data(), bsr_diag_val.data(),
                   &P->d_Aii_rowptr, &P->d_Aii_colind, &P->d_Aii_val) != 0) {
        return -1;
    }
    
    // Extract off-diagonal part (A_ij)
    int* offd_i = hypre_CSRMatrixI(offd);
    int* offd_j = hypre_CSRMatrixJ(offd);
    double* offd_data = hypre_CSRMatrixData(offd);
    int offd_nnz = hypre_CSRMatrixNumNonzeros(offd);
    
    P->Aij_nnzb = offd_nnz;
    P->nb_ghost_host = offd_cols;
    P->nb_ghost = offd_cols;
    
    if (offd_nnz > 0) {
        std::vector<int> bsr_offd_rowptr(local_rows + 1);
        std::vector<int> bsr_offd_colind(offd_nnz);
        std::vector<float> bsr_offd_val(offd_nnz);
        
        for (int i = 0; i <= local_rows; i++) {
            bsr_offd_rowptr[i] = offd_i[i];
        }
        
        for (int k = 0; k < offd_nnz; k++) {
            bsr_offd_colind[k] = offd_j[k];  // Local ghost column indices
            bsr_offd_val[k] = (float)offd_data[k];
        }
        
        // Upload off-diagonal part to device
        if (upload_bsr(local_rows, P->Aij_nnzb, bdim,
                       bsr_offd_rowptr.data(), bsr_offd_colind.data(), bsr_offd_val.data(),
                       &P->d_Aij_rowptr, &P->d_Aij_colind, &P->d_Aij_val) != 0) {
            return -1;
        }
        
        // Extract global column map for ghost columns
        HYPRE_BigInt* col_map_offd = hypre_ParCSRMatrixColMapOffd(par_A);
        P->h_col_map_offd = (gblk*)malloc(offd_cols * sizeof(gblk));
        if (!P->h_col_map_offd) return -1;
        
        for (int i = 0; i < offd_cols; i++) {
            P->h_col_map_offd[i] = (gblk)col_map_offd[i];
        }
    } else {
        // No off-diagonal entries
        P->d_Aij_rowptr = nullptr;
        P->d_Aij_colind = nullptr;
        P->d_Aij_val = nullptr;
        P->h_col_map_offd = nullptr;
    }
    
    // Allocate vectors
    int n_local = P->mb_local * P->bdim;
    int n_ghost = P->nb_ghost * P->bdim;
    
    CHECK_CUDA(cudaMalloc((void**)&P->d_x_local, n_local * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&P->d_x_ghost, 
                         (n_ghost > 0 ? n_ghost : 1) * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&P->d_y, n_local * sizeof(float)));
    
    // Initialize vectors to zero
    CHECK_CUDA(cudaMemset(P->d_x_local, 0, n_local * sizeof(float)));
    CHECK_CUDA(cudaMemset(P->d_x_ghost, 0, (n_ghost > 0 ? n_ghost : 1) * sizeof(float)));
    CHECK_CUDA(cudaMemset(P->d_y, 0, n_local * sizeof(float)));
    
    // Create modern cuSPARSE API descriptors
    if (create_cusparse_descriptors(P) != 0) {
        return -1;
    }
    
    return 0;
}

double parbsr_verify_spmv(const ParBSR* P,
                          HypreParMatrix* hypre_A,
                          const float* x_data, int x_size,
                          const float* y_our_data, int y_size)
{
    if (!P || !hypre_A || !x_data || !y_our_data) return -1.0;
    
    // Create MFEM vectors for Hypre SpMV
    Vector x_hypre(x_size), y_hypre(y_size);
    
    // Copy input vector (convert float to double)
    for (int i = 0; i < x_size; i++) {
        x_hypre[i] = (double)x_data[i];
    }
    
    // Perform Hypre SpMV
    hypre_A->Mult(x_hypre, y_hypre);
    
    // Compare results
    double max_diff = 0.0;
    for (int i = 0; i < y_size; i++) {
        double diff = std::abs(y_hypre[i] - (double)y_our_data[i]);
        max_diff = std::max(max_diff, diff);
    }
    
    return max_diff;
}