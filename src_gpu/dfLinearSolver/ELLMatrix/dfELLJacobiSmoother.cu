#include "dfELLSmoother.H"

__global__ void ellJacobiSmooth
(
    int nCells,
    double* psi,
    double* psiCopyPtr,
    double* source,
    int ell_row_maxcount,
    int* d_ell_cols, 
    double* d_ell_values,
    double* diagPtr
)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nCells)
        return;
    
    double sum = source[index];
    int offset = ell_row_maxcount * index;
    for(int r = 0; r < ell_row_maxcount; r++){
        sum -= d_ell_values[r + offset] * psiCopyPtr[d_ell_cols[r + offset]];
    }
    psi[index] = sum / diagPtr[index];
}

void ELLJacobiSmoother::smooth
(
    cudaStream_t stream,
    int nSweeps,
    int nCells,
    double* psi,
    double* source,
    int ell_row_maxcount,
    int* d_ell_cols, 
    double* d_ell_values,
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
        ellJacobiSmooth<<<blocks_per_grid, threads_per_block, 0, stream>>>
            (nCells, psi, psiCopyPtr, source, ell_row_maxcount, d_ell_cols, d_ell_values, diagPtr);

    }
};