#include "dfELLSmoother.H"
#include "dfSolverOpBase.H"

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
    double* diagPtr,
    // PARALLEL_
    const dfMatrixDataBase& dataBase,
    double* scalarSendBufList_, 
    double* scalarRecvBufList_,
    double** interfaceBouCoeffs,
    int** faceCells, std::vector<int> nPatchFaces
)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (nCells + threads_per_block - 1) / threads_per_block;
    
    double* bPrime;
    cudaMalloc(&bPrime, nCells * sizeof(double));

    for (int sweep=0; sweep<nSweeps; sweep++)
    {
        cudaMemcpyAsync(bPrime, source, nCells * sizeof(double), cudaMemcpyDeviceToDevice, stream);

        // // for debug start
        // checkCudaErrors(cudaStreamSynchronize(stream));
        // double source_sum = 0.0;
        // double* test_result;
        // cudaMalloc(&test_result, sizeof(double));
        // reduce(nCells, threads_per_block, blocks_per_grid, bPrime, test_result, dataBase.stream, false);
        // #ifndef PARALLEL_
        //     cudaMemcpyAsync(&source_sum, &test_result[0] , sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
        // #else
        //     ncclAllReduce(&test_result[0], &test_result[0], 1, ncclDouble, ncclSum, dataBase.nccl_comm, dataBase.stream);
        //     cudaStreamSynchronize(dataBase.stream);
        //     cudaMemcpyAsync(&source_sum, &test_result[0], sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
        // #endif
        // std::cout << nCells << " **gpu source_sum in smooth before: " << source_sum << std::endl;
        // // for debug end

#ifdef PARALLEL_   
        // sign = -1 for negate()
        // --- initMatrixInterfaces & updateMatrixInterfaces ---
        updateMatrixInterfaceCoeffs(
            dataBase.stream, dataBase.neighbProcNo, dataBase.nccl_comm,
            nPatchFaces, psi, bPrime, 
            scalarSendBufList_, scalarRecvBufList_,
            interfaceBouCoeffs, faceCells, -1.0);
#endif
        // // for debug start
        // reduce(nCells, threads_per_block, blocks_per_grid, bPrime, test_result, dataBase.stream, false);
        // #ifndef PARALLEL_
        //     cudaMemcpyAsync(&source_sum, &test_result[0] , sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
        // #else
        //     ncclAllReduce(&test_result[0], &test_result[0], 1, ncclDouble, ncclSum, dataBase.nccl_comm, dataBase.stream);
        //     cudaStreamSynchronize(dataBase.stream);
        //     cudaMemcpyAsync(&source_sum, &test_result[0], sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
        // #endif
        // std::cout << nCells << " **gpu source_sum in smooth: " << source_sum << std::endl;
        // // for debug end

        double* psiCopyPtr;
        cudaMalloc(&psiCopyPtr, nCells * sizeof(double));
        cudaMemcpyAsync(psiCopyPtr, psi, nCells * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    
        size_t threads_per_block = 1024;
        size_t blocks_per_grid = (nCells + threads_per_block - 1) / threads_per_block;
        ellJacobiSmooth<<<blocks_per_grid, threads_per_block, 0, stream>>>
            (nCells, psi, psiCopyPtr, bPrime, ell_row_maxcount, d_ell_cols, d_ell_values, diagPtr);
        checkCudaErrors(cudaStreamSynchronize(stream));

        // // for debug start
        // reduce(nCells, threads_per_block, blocks_per_grid, psi, test_result, dataBase.stream, false);
        // #ifndef PARALLEL_
        //     cudaMemcpyAsync(&source_sum, &test_result[0] , sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
        // #else
        //     ncclAllReduce(&test_result[0], &test_result[0], 1, ncclDouble, ncclSum, dataBase.nccl_comm, dataBase.stream);
        //     cudaStreamSynchronize(dataBase.stream);
        //     cudaMemcpyAsync(&source_sum, &test_result[0], sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
        // #endif
        // std::cout << nCells << " **gpu source_sum psi: " << source_sum << std::endl;
        // // for debug end
    }
};