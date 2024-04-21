#include "dfCSRSolver.H"
#include "dfSolverOpBase.H"
#include "dfMatrixDataBase.H"

#define PARALLEL_
#define PRINT_

// kernel functions for PBiCGStab solver

void PCGCSRSolver::initialize(const int nCells, const size_t boundary_surface_value_bytes)
{
    // cudamalloc variables related to PCGSolver
    cudaMalloc(&d_wA, nCells * sizeof(double));
    cudaMalloc(&d_rA, nCells * sizeof(double));
    cudaMalloc(&reduce_result, sizeof(double));
    // for parallel
    cudaMalloc(&scalarSendBufList_, boundary_surface_value_bytes);
    cudaMalloc(&scalarRecvBufList_, boundary_surface_value_bytes);

    // preconditioner
    precond_ = new GAMGCSRPreconditioner();
    precond_->initialize();
}

void PCGCSRSolver::solve
(
    const dfMatrixDataBase& dataBase,
    const double* d_internal_coeffs,
    const double* d_boundary_coeffs,
    int* patch_type,
    double* diagPtr,
    const double* off_diag_value,
    const double *rhs, 
    double *psi
)
{
    printf("GPU-CSR-PBiCGStab::solve start --------------------------------------------\n");

    int nIterations = 0;
 
    const int row_ = dataBase.num_total_cells;
    const int nCells = dataBase.num_cells;

    // these two int control reduce's scale
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (nCells + threads_per_block - 1) / threads_per_block;

#ifdef PRINT_    
    printf("threads_per_block = %d, blocks_per_grid = %d\n",threads_per_block, blocks_per_grid);
#endif

    double psi_ave = 0.;
    double normFactor = 0.;
    double initialResidual = 0.;
    double finalResidual = 0.;

    // --- reduce psi to get : psi_ave ---
    reduce(nCells, threads_per_block, blocks_per_grid, psi, &psi_ave, dataBase.stream, false);
#ifdef PARALLEL_
    ncclAllReduce(&psi_ave, &psi_ave, 1, ncclDouble, ncclSum, dataBase.nccl_comm, dataBase.stream);
    cudaStreamSynchronize(dataBase.stream);
#endif
    psi_ave = psi_ave / row_;

#ifdef PRINT_
    printf("psi_ave = %.10e\n",psi_ave);
#endif

    // --- addInternalCoeffs : diag ---
    // input : d_internal_coeffs
    addInternalCoeffs(dataBase.stream, dataBase.num_patches, dataBase.patch_size, 
        d_internal_coeffs, dataBase.d_boundary_face_cell, diagPtr, patch_type);
    
    // --- SpMV : wA ---
    // input : psi, diag
    SpMV4CSR(dataBase.stream, nCells, diagPtr, off_diag_value, dataBase.d_csr_row_index_no_diag, dataBase.d_csr_col_index_no_diag, psi, d_wA); 

#ifdef PARALLEL_
    // --- initMatrixInterfaces & updateMatrixInterfaces : wA ---
    // input : psi (neighbor's psi)
    updateMatrixInterfaces(
        dataBase.stream, dataBase.num_patches, dataBase.patch_size,
        dataBase.neighbProcNo, dataBase.nccl_comm,
        dataBase.interfaceFlag, psi, d_wA, 
        scalarSendBufList_, scalarRecvBufList_,
        d_boundary_coeffs, dataBase.d_boundary_face_cell, patch_type);
#endif


};