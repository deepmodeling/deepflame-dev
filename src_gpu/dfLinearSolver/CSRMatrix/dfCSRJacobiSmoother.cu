#include "dfCSRSmoother.H"

__global__ void csrJacobiSmooth
(
    int nCells,
    double* psi,
    double* psiCopyPtr,
    double* source,
    double* off_diag_value_Ptr,
    int* off_diag_rowptr_Ptr, 
    int* off_diag_colidx_Ptr,
    double* diagPtr
)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nCells)
        return;
    
    double sum = source[index];
    for(int r = off_diag_rowptr_Ptr[index]; r < off_diag_rowptr_Ptr[index + 1]; r++){
        sum -= off_diag_value_Ptr[r] * psiCopyPtr[off_diag_colidx_Ptr[r]];
    }
    psi[index] = sum / diagPtr[index];
}

void CSRJacobiSmoother::smooth
(
    cudaStream_t stream,
    int nSweeps,
    int nCells,
    double* psi,
    double* source,
    double* off_diag_value_Ptr,
    int* off_diag_rowptr_Ptr, 
    int* off_diag_colidx_Ptr,
    double* diagPtr
)
{
    for (int sweep=0; sweep<nSweeps; sweep++)
    {

        double* psiCopyPtr;
        cudaMalloc(&psiCopyPtr, nCells * sizeof(double));
        cudaMemcpyAsync(psiCopyPtr, psi, nCells * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    
        size_t threads_per_block = 1024;
        size_t blocks_per_grid = (nCells + threads_per_block - 1) / threads_per_block;
        csrJacobiSmooth<<<blocks_per_grid, threads_per_block, 0, stream>>>
            (nCells, psi, psiCopyPtr, source, off_diag_value_Ptr, off_diag_rowptr_Ptr, off_diag_colidx_Ptr, diagPtr);

    }
};