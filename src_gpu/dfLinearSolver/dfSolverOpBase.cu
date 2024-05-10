#include "dfSolverOpBase.H"
#include "dfMatrixDataBase.H"
#include "dfNcclBase.H"

bool isPow2(int n) {
    if (n <= 0) {
        return false;
    }

    return (n & (n - 1)) == 0;
}

// --- kernel ----

__global__ void kernel_addInternalCoeffs
(
    int num, 
    int offset, 
    const double* internal_coeffs, 
    int* face2Cells, 
    double* diagPtr
)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num)
        return;

    int start_index = offset + index;
    int cellIndex = face2Cells[start_index];
    diagPtr[cellIndex] += internal_coeffs[start_index];
}

__global__ void kernel_SpMV_csr
(
    int nCells,
    double* diagPtr, 
    const double* off_diag_value_Ptr,
    int* off_diag_rowptr_Ptr,
    int* off_diag_colidx_Ptr,
    double* input,
    double* output
)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nCells)
        return;

    double tmp = diagPtr[index] * input[index];
    for(int r = off_diag_rowptr_Ptr[index]; r < off_diag_rowptr_Ptr[index+1]; r++){
        tmp += off_diag_value_Ptr[r] * input[off_diag_colidx_Ptr[r]];
    }
    output[index] = tmp;
}
__global__ void kernel_SpMV_ell
(
    int nCells,
    double* diagPtr, 
    double* ellValues,
    int* ellCols,
    int ell_max_count_,
    double* input,
    double* output
)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nCells)
        return;

    double tmp = diagPtr[index] * input[index];
    int offset = ell_max_count_ * index;
    for(int r = 0; r < ell_max_count_; r++){
        double value = ellValues[r + offset];
        int col = ellCols[r + offset];
        tmp += value * input[col];
    }
    output[index] = tmp;
}

__global__ void kernel_2_csr
(
    int nCells,
    double* rAPtr,
    const double* source,
    double* yAPtr,
    double* diagPtr,
    const double* off_diag_value_Ptr,
    int* off_diag_rowptr_Ptr,
    double* pAPtr
)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nCells)
        return;

    rAPtr[index] = source[index] - yAPtr[index];

    double tmp = diagPtr[index];
    for (int r = off_diag_rowptr_Ptr[index]; r < off_diag_rowptr_Ptr[index + 1]; r++){
        tmp += off_diag_value_Ptr[r];
    }
    pAPtr[index] = tmp;
}
__global__ void kernel_2_ell
(
    int nCells,
    double* rAPtr,
    const double* source,
    double* yAPtr,
    double* diagPtr,
    double* ellValues,
    int ell_max_count_,
    double* pAPtr
)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nCells)
        return;

    rAPtr[index] = source[index] - yAPtr[index];

    double tmp = diagPtr[index];
    int offset = ell_max_count_ * index;
    for(int r = 0; r < ell_max_count_; r++){
        tmp += ellValues[r + offset];
    }
    pAPtr[index] = tmp;

}

__global__ void kernel_subBoundaryCoeffs
(
    int num, 
    int offset, 
    const double* boundary_coeffs, 
    int* face2Cells, 
    double* pAPtr
)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num)
        return;

    int start_index = offset + index;
    int cellIndex = face2Cells[start_index];
    pAPtr[cellIndex] -= boundary_coeffs[start_index];
}

__global__ void kernel_4
(
    int nCells,
    double psi_ave,
    double* pAPtr,
    double* normFactor,
    double* yAPtr,
    const double* source
)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nCells)
        return;
    
    pAPtr[index] *= psi_ave;
    normFactor[index] = std::abs(yAPtr[index] - pAPtr[index]) + std::abs(source[index] - pAPtr[index]);
}

__global__ void kernel_AmulBtoC(int nCells, double *input1, double *input2, double* output)
{

    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nCells)
        return;

    output[index] = input1[index] * input2[index]; 
}
__global__ void kernel_AmulAtoB(int nCells, double *input, double* output)
{

    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nCells)
        return;

    output[index] = input[index] * input[index]; 
}


__global__ void kernel_6_zero
(
    int nCells,
    double* pAPtr,
    double* rAPtr,
    double* yAPtr
)
{

    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nCells)
        return;
    
    pAPtr[index] = rAPtr[index]; 
    yAPtr[index] = pAPtr[index];
}
__global__ void kernel_6
(
    int nCells,
    double* pAPtr,
    double* rAPtr,
    double beta,
    double omega,
    double* AyAPtr,
    double* yAPtr
)
{

    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nCells)
        return;
    
    pAPtr[index] = rAPtr[index] + beta*(pAPtr[index] - omega*AyAPtr[index]); 
    yAPtr[index] = pAPtr[index];
}

__global__ void kernel_7(int nCells, double *sAPtr, double *rAPtr, double alpha, double *AyAPtr)
{

    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nCells)
        return;

    sAPtr[index] = rAPtr[index] - alpha * AyAPtr[index];
}


__global__ void kernel_AmulBaddtoC(int nCells, double input1, double *input2, double* output)
{

    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nCells)
        return;

    output[index] += input1 * input2[index]; 
}

__global__ void kernel_9(int nCells, double *psiPtr, double *yAPtr, 
        double *zAPtr, double *rAPtr, double *sAPtr, double *tAPtr, 
        double alpha, double omega)
{

    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nCells)
        return;

    psiPtr[index] += alpha * yAPtr[index] + omega * zAPtr[index];

    rAPtr[index] = sAPtr[index] - omega * tAPtr[index];
}

__global__ void kernel_initMatrixInterfaces
(
    double* input, 
    int interfaceiSize,
    double* scalarSendBufList_,
    int offset,
    int* face2Cells
)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= interfaceiSize)
        return;

    int start_index = offset + index;
    int cellIndex = face2Cells[start_index];
    scalarSendBufList_[start_index] = input[cellIndex];
}

__global__ void kernel_updateMatrixInterfaces
(
    const double* d_boundary_coeffs,
    int interfaceiSize,
    double* scalarRecvBufList_,
    int offset,
    int* face2Cells,
    double* output
)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= interfaceiSize)
        return;

    int start_index = offset + index;
    int cellIndex = face2Cells[start_index];
    output[cellIndex] -= d_boundary_coeffs[start_index] * scalarRecvBufList_[start_index];
}

// PCG
__global__ void kernel_10
(
    int nCells,
    double* pAPtr,
    double* wAPtr,
    double beta
)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nCells)
        return;

    pAPtr[index] = wAPtr[index] + beta*pAPtr[index];
}

__global__ void kernel_11
(
    int nCells,
    double* psiPtr,
    double* pAPtr,
    double* rAPtr,
    double* wAPtr,
    double alpha
){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nCells)
        return;

    psiPtr[index] += alpha*pAPtr[index];
    rAPtr[index] -= alpha*wAPtr[index];
}


// --- member functions ---

void addInternalCoeffs(
        cudaStream_t stream, int num_patches, std::vector<int> patch_size, 
        const double *d_internal_coeffs, int *d_boundary_face_cell, 
        double *diagPtr, int *patch_type) 
{
    int offset = 0;
    for (int i = 0; i < num_patches; i++) {
        if (patch_size[i] == 0) continue;
        size_t threads_per_block = 1024;
        size_t blocks_per_grid = (patch_size[i] + threads_per_block - 1) / threads_per_block;
        kernel_addInternalCoeffs<<<blocks_per_grid, threads_per_block, 0, stream>>>
            (patch_size[i], offset, d_internal_coeffs, d_boundary_face_cell, diagPtr);
        (patch_type[i] == boundaryConditions::processor
            || patch_type[i] == boundaryConditions::processorCyclic) ?
            offset += 2 * patch_size[i] : offset += patch_size[i];
    }
}

void SpMV4CSR(
        cudaStream_t stream, const int nCells, double *diagPtr, const double *off_diag_value,
        int *d_csr_row_index_no_diag, int *d_csr_col_index_no_diag, double *input, double *output)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (nCells + threads_per_block - 1) / threads_per_block;

    kernel_SpMV_csr<<<blocks_per_grid, threads_per_block, 0, stream>>>
        (nCells, diagPtr, off_diag_value, d_csr_row_index_no_diag, 
        d_csr_col_index_no_diag, input, output);
}
void SpMV4ELL(
        cudaStream_t stream, const int nCells, double *diagPtr, double *ellValues, 
        int *ellCols, int ell_max_count_, double *input, double *output)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (nCells + threads_per_block - 1) / threads_per_block;

    kernel_SpMV_ell<<<blocks_per_grid, threads_per_block, 0, stream>>>
        (nCells, diagPtr, ellValues, ellCols, ell_max_count_, input, output);
}

void calrAandpA4CSR(
        cudaStream_t stream, const int nCells, double *d_rA, const double *rhs, double *d_yA,
        double *diagPtr, const double *off_diag_value, int *d_csr_row_index_no_diag, double *d_pA)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (nCells + threads_per_block - 1) / threads_per_block;

    kernel_2_csr<<<blocks_per_grid, threads_per_block, 0, stream>>>
        (nCells, d_rA, rhs, d_yA, diagPtr, off_diag_value, d_csr_row_index_no_diag, d_pA);
}

void calrAandpA4ELL(
        cudaStream_t stream, const int nCells, double *d_rA, const double *rhs, 
        double *d_yA, double *diagPtr, double *ellValues, int ell_max_count_, double *d_pA)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (nCells + threads_per_block - 1) / threads_per_block;

    kernel_2_ell<<<blocks_per_grid, threads_per_block, 0, stream>>>
        (nCells, d_rA, rhs, d_yA, diagPtr, ellValues, ell_max_count_, d_pA);
}


void subBoundaryCoeffs(
        cudaStream_t stream, int num_patches, std::vector<int> patch_size, 
        const double *d_boundary_coeffs, int *d_boundary_face_cell, 
        double *d_pA, int *patch_type)
{
    int offset = 0;
    for (int i = 0; i < num_patches; i++) {
        if (patch_size[i] == 0) continue;
        size_t threads_per_block = 1024;
        size_t blocks_per_grid = (patch_size[i] + threads_per_block - 1) / threads_per_block;
        kernel_subBoundaryCoeffs<<<blocks_per_grid, threads_per_block, 0, stream>>>
            (patch_size[i], offset, d_boundary_coeffs, d_boundary_face_cell, d_pA);
        (patch_type[i] == boundaryConditions::processor
            || patch_type[i] == boundaryConditions::processorCyclic) ?
            offset += 2 * patch_size[i] : offset += patch_size[i];
    }
}

void calpAandnormFactor(
        cudaStream_t stream,const int nCells, double psi_ave, double* d_pA,
        double* d_normFactors_tmp, double* d_yA, const double* rhs)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (nCells + threads_per_block - 1) / threads_per_block;

    kernel_4<<<blocks_per_grid, threads_per_block, 0, stream>>>
        (nCells, psi_ave, d_pA, d_normFactors_tmp, d_yA, rhs);
}

void AmulBtoC(
        cudaStream_t stream, int nCells, double *input1, double *input2, double *output)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (nCells + threads_per_block - 1) / threads_per_block;

    kernel_AmulBtoC<<<blocks_per_grid, threads_per_block, 0, stream>>>
        (nCells, input1, input2, output);
}

void calpAandyAInit(
        cudaStream_t stream, int nCells, double *d_pA, double *d_rA, double *d_yA)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (nCells + threads_per_block - 1) / threads_per_block;

    kernel_6_zero<<<blocks_per_grid, threads_per_block, 0, stream>>>
        (nCells, d_pA, d_rA, d_yA);

}

void calpAandyA(
        cudaStream_t stream, int nCells, double *d_pA, double *d_rA,
        double beta, double omega, double *d_AyA, double *d_yA)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (nCells + threads_per_block - 1) / threads_per_block;

    kernel_6<<<blocks_per_grid, threads_per_block, 0, stream>>>
        (nCells, d_pA, d_rA, beta, omega, d_AyA, d_yA);
}

void calsA(
    cudaStream_t stream, int nCells, double *d_sA, 
    double *d_rA, double alpha, double *d_AyA
){
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (nCells + threads_per_block - 1) / threads_per_block;

    kernel_7<<<blocks_per_grid, threads_per_block, 0, stream>>>
        (nCells, d_sA, d_rA, alpha, d_AyA);
}

void exitLoop(
        cudaStream_t stream, int nCells, double input1, double *input2, double* output)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (nCells + threads_per_block - 1) / threads_per_block;

    kernel_AmulBaddtoC<<<blocks_per_grid, threads_per_block, 0, stream>>>
        (nCells, input1, input2, output);
}

void AmulAtoB(
    cudaStream_t stream, int nCells, double *input, double* output
){
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (nCells + threads_per_block - 1) / threads_per_block;

    kernel_AmulAtoB<<<blocks_per_grid, threads_per_block, 0, stream>>>
        (nCells, input, output);
}

void calpsiandrA(
    cudaStream_t stream, int nCells, double *psi, double *d_yA, double *d_zA, 
    double *d_rA, double *d_sA, double *d_tA, double alpha, double omega
){
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (nCells + threads_per_block - 1) / threads_per_block;

    kernel_9<<<blocks_per_grid, threads_per_block, 0, stream>>>
        (nCells, psi, d_yA, d_zA, d_rA, d_sA, d_tA, alpha, omega);
}

void updateMatrixInterfaces(
    cudaStream_t stream, int num_patches, std::vector<int> patch_size,
    std::vector<int> neighbProcNo,  ncclComm_t nccl_comm,
    int *interfaceFlag, double *input, double *output, 
    double *scalarSendBufList_, double *scalarRecvBufList_,
    const double *d_boundary_coeffs, int *d_boundary_face_cell, int *patch_type
){
    int offset = 0;
    for (int i = 0; i < num_patches; i++) {
        if (patch_size[i] == 0) continue;
        else if (interfaceFlag[i] == 0){
            (patch_type[i] == boundaryConditions::processor
                || patch_type[i] == boundaryConditions::processorCyclic) ?
                offset += 2 * patch_size[i] : offset += patch_size[i];
            continue;
        }
        size_t threads_per_block = 1024;
        size_t blocks_per_grid = (patch_size[i] + threads_per_block - 1) / threads_per_block;
        kernel_initMatrixInterfaces<<<blocks_per_grid, threads_per_block, 0, stream>>>
            (input, patch_size[i], scalarSendBufList_, offset, d_boundary_face_cell);
        ncclGroupStart();
        ncclSend(scalarSendBufList_ + offset, patch_size[i], ncclDouble, neighbProcNo[i], nccl_comm, stream);
        ncclRecv(scalarRecvBufList_ + offset, patch_size[i], ncclDouble, neighbProcNo[i], nccl_comm, stream);
        ncclGroupEnd();
        threads_per_block = 1024;
        blocks_per_grid = (patch_size[i] + threads_per_block - 1) / threads_per_block;
        kernel_updateMatrixInterfaces<<<blocks_per_grid, threads_per_block, 0, stream>>>
            (d_boundary_coeffs, patch_size[i], scalarRecvBufList_, offset, d_boundary_face_cell, output);
        (patch_type[i] == boundaryConditions::processor
            || patch_type[i] == boundaryConditions::processorCyclic) ?
            offset += 2 * patch_size[i] : offset += patch_size[i];
    }

}

// PCG
void calpA(
    cudaStream_t stream, int nCells, double* pAPtr, double* wAPtr, double beta
){
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (nCells + threads_per_block - 1) / threads_per_block;

    kernel_10<<<blocks_per_grid, threads_per_block, 0, stream>>>
        (nCells, pAPtr, wAPtr, beta);
}

void calpsiandrA(
    cudaStream_t stream,  int nCells, double* psi, double* pA, double* rA, double* wA, double alpha
){
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (nCells + threads_per_block - 1) / threads_per_block;

    kernel_11<<<blocks_per_grid, threads_per_block, 0, stream>>>
        (nCells, psi, pA, rA, wA, alpha);
}


// ---------------------------- reduce -------------------------------

template <class T>
__device__ __forceinline__ T warpReduceSum(unsigned int mask, T mySum) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    mySum += __shfl_down_sync(mask, mySum, offset);
  }
  return mySum;
}

template <class T>
struct SharedMemory {
  __device__ inline operator T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() const {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};


template <>
struct SharedMemory<double> {
  __device__ inline operator double *() {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }

  __device__ inline operator const double *() const {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }
};

template <typename T, unsigned int blockSize, bool nIsPow2>
__global__ void reduce7(const T *__restrict__ g_idata, T *__restrict__ g_odata,
                        unsigned int n, bool isabs) {
  T *sdata = SharedMemory<T>();

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int gridSize = blockSize * gridDim.x;
  unsigned int maskLength = (blockSize & 31);  // 31 = warpSize-1
  maskLength = (maskLength > 0) ? (32 - maskLength) : maskLength;
  const unsigned int mask = (0xffffffff) >> maskLength;

  T mySum = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  if (nIsPow2) {
    unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
    gridSize = gridSize << 1;

    while (i < n) {
      if(isabs)
        mySum += std::abs(g_idata[i]);
      else
        mySum += g_idata[i];
      // ensure we don't read out of bounds -- this is optimized away for
      // powerOf2 sized arrays
      if ((i + blockSize) < n) {
        if(isabs)
            mySum += std::abs(g_idata[i + blockSize]);
        else
            mySum += g_idata[i + blockSize];
      }
      i += gridSize;
    }
  } else {
    unsigned int i = blockIdx.x * blockSize + threadIdx.x;
    while (i < n) {
      if(isabs)
        mySum += std::abs(g_idata[i]);
      else
        mySum += g_idata[i];
      i += gridSize;
    }
  }

  // Reduce within warp using shuffle or reduce_add if T==int & CUDA_ARCH ==
  // SM 8.0
  mySum = warpReduceSum<T>(mask, mySum);

  // each thread puts its local sum into shared memory
  if ((tid % warpSize) == 0) {
    sdata[tid / warpSize] = mySum;
  }

  __syncthreads();

  const unsigned int shmem_extent =
      (blockSize / warpSize) > 0 ? (blockSize / warpSize) : 1;
  const unsigned int ballot_result = __ballot_sync(mask, tid < shmem_extent);
  if (tid < shmem_extent) {
    mySum = sdata[tid];
    // Reduce final warp using shuffle or reduce_add if T==int & CUDA_ARCH ==
    // SM 8.0
    mySum = warpReduceSum<T>(ballot_result, mySum);
  }

  // write result for this block to global mem
  if (tid == 0) {
    // g_odata[blockIdx.x] = mySum;
    atomicAdd(&(g_odata[0]), mySum);
  }
}

template <class T>
void runReduce(int size, int threads, int blocks, double *d_idata, double *d_odata, cudaStream_t stream, const bool isabs)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    int smemSize = ((threads / 32) + 1) * sizeof(T);
    if (isPow2(size)) {
        switch (threads) {
          case 1024:
            reduce7<T, 1024, true>
                <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size, isabs);
            break;

          case 512:
            reduce7<T, 512, true>
                <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size, isabs);
            break;

          case 256:
            reduce7<T, 256, true>
                <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size, isabs);
            break;

          case 128:
            reduce7<T, 128, true>
                <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size, isabs);
            break;

          case 64:
            reduce7<T, 64, true>
                <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size, isabs);
            break;

          case 32:
            reduce7<T, 32, true>
                <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size, isabs);
            break;

          case 16:
            reduce7<T, 16, true>
                <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size, isabs);
            break;

          case 8:
            reduce7<T, 8, true>
                <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size, isabs);
            break;

          case 4:
            reduce7<T, 4, true>
                <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size, isabs);
            break;

          case 2:
            reduce7<T, 2, true>
                <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size, isabs);
            break;

          case 1:
            reduce7<T, 1, true>
                <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size, isabs);
            break;
        }
    } else {
        switch (threads) {
          case 1024:
            reduce7<T, 1024, false>
                <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size, isabs);
            break;
          case 512:
            reduce7<T, 512, false>
                <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size, isabs);
            break;

          case 256:
            reduce7<T, 256, false>
                <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size, isabs);
            break;

          case 128:
            reduce7<T, 128, false>
                <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size, isabs);
            break;

          case 64:
            reduce7<T, 64, false>
                <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size, isabs);
            break;

          case 32:
            reduce7<T, 32, false>
                <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size, isabs);
            break;

          case 16:
            reduce7<T, 16, false>
                <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size, isabs);
            break;

          case 8:
            reduce7<T, 8, false>
                <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size, isabs);
            break;

          case 4:
            reduce7<T, 4, false>
                <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size, isabs);
            break;

          case 2:
            reduce7<T, 2, false>
                <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size, isabs);
            break;

          case 1:
            reduce7<T, 1, false>
                <<<dimGrid, dimBlock, smemSize, stream>>>(d_idata, d_odata, size, isabs);
            break;
        }
    }
}

void reduce(int size, int threads, int blocks, double *d_idata, 
    double *d_odata, cudaStream_t stream, const bool isabs)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

    cudaMemset(d_odata, 0 ,sizeof(double));
    runReduce<double>(size, threads, blocks, d_idata, d_odata, stream, isabs);
}

/*==========================================================================*/
/*=============================GAMG Start===================================*/
__global__ void kernel_restrict
(
    int nFineCells,
    int* d_restrictMap,
    double* fineField,
    double* coarseField
)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nFineCells)
        return;

    int mapIndex = d_restrictMap[index];
    if(mapIndex >= 0)
    {
        atomicAdd(&coarseField[mapIndex], fineField[index]);
    }
}

__global__ void kernel_restrictMatrix
(
    int nFineFaces, int* d_faceRestrictMap, int* d_faceFlipMap,
    double* d_fineUpper, double* d_fineLower,
    double* d_coarseUpper, double* d_coarseLower, double* d_coarseDiag
)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nFineFaces)
        return;

    int faceMapIndex = d_faceRestrictMap[index];
    if (faceMapIndex >= 0) 
    {
        if (d_faceFlipMap[index] > 0)
        {
            atomicAdd(&d_coarseUpper[faceMapIndex], d_fineLower[index]);
            atomicAdd(&d_coarseLower[faceMapIndex], d_fineUpper[index]);
        }
        else
        {
            atomicAdd(&d_coarseUpper[faceMapIndex], d_fineUpper[index]);
            atomicAdd(&d_coarseLower[faceMapIndex], d_fineLower[index]);
        }
    }
    else
    {
        int diagIndex = -1 - faceMapIndex;
        atomicAdd(&d_coarseDiag[diagIndex], (d_fineUpper[index] + d_fineLower[index]));
    }
}

__global__ void kernel_prolong
(
    int nFineCells,
    int* d_restrictMap,
    double* fineField,
    double* coarseField
)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nFineCells)
        return;

    int mapIndex = d_restrictMap[index];
    fineField[index] = coarseField[mapIndex];
}

__global__ void kernel_calcNumDenom
(
    int nCells,
    double* field, double* source, double* Acf,
    double* scalingFactorNum, double* scalingFactorDenom
)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nCells)
        return;
    
    scalingFactorNum[index] = source[index] * field[index];
    scalingFactorDenom[index] = Acf[index] * field[index];
}

__global__ void kernel_scale
(
    int nCells, double scalingFactor,
    double* field, double* source, double* Acf, double* diag
)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nCells)
        return;
    
    field[index] = scalingFactor*field[index] 
                    + (source[index] - scalingFactor*Acf[index]) / diag[index];
}

__global__ void kernel_updateSource
(
    int nCells,
    double* field, double* Acf
)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nCells)
        return;
    
    atomicAdd(&field[index], -Acf[index]);
}

__global__ void kernel_updateCorr
(
    int nCells,
    double* field, double* preSmooth
)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nCells)
        return;
    
    atomicAdd(&field[index], preSmooth[index]);
}

__global__ void kernel_directSolve1x1
(
    int nCells,
    double* d_diag, double* d_corr, double* d_source
)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index == 0) 
    { 
        d_corr[index] = d_source[index] / d_diag[index]; 
    }
}

void restrictFieldGPU(cudaStream_t stream, int nFineCells, int* d_restrictMap, 
                        double* d_fineField, double* d_coarseField)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (nFineCells + threads_per_block - 1) / threads_per_block;

    kernel_restrict<<<blocks_per_grid, threads_per_block, 0, stream>>>
        (nFineCells, d_restrictMap, d_fineField, d_coarseField);
    checkCudaErrors(cudaStreamSynchronize(stream));
}

void restrictMatrixGPU(cudaStream_t stream, int nFineFaces, int* d_faceRestrictMap, int* d_faceFlipMap,
                        double* d_fineUpper, double* d_fineLower, 
                        double* d_coarseUpper, double* d_coarseLower, double* d_coarseDiag)
{
    // TODO: if matrix is symmetric, kernel can be simplified, not implement yet.
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (nFineFaces + threads_per_block - 1) / threads_per_block;

    kernel_restrictMatrix<<<blocks_per_grid, threads_per_block, 0, stream>>>
        (nFineFaces, d_faceRestrictMap, d_faceFlipMap, d_fineUpper, d_fineLower, 
        d_coarseUpper, d_coarseLower, d_coarseDiag);
    checkCudaErrors(cudaStreamSynchronize(stream));
}

void prolongFieldGPU(cudaStream_t stream, int nFineCells, int* d_restrictMap, 
                        double* d_fineField, double* d_coarseField)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (nFineCells + threads_per_block - 1) / threads_per_block;

    kernel_prolong<<<blocks_per_grid, threads_per_block, 0, stream>>>
        (nFineCells, d_restrictMap, d_fineField, d_coarseField);
    checkCudaErrors(cudaStreamSynchronize(stream));
}

void scaleFieldGPU( const dfMatrixDataBase& dataBase, int nCells, 
                    double* d_Field, double* d_Source, double* d_AcfField, 
                    double* diag, double* off_diag_value,
                    int* csr_row_index_no_diag, int* csr_col_index_no_diag, 
                    double** interfaceIntCoeffs, double** interfaceBouCoeffs,
                    int** faceCells, std::vector<int> nPatchFaces, 
                    double* d_scalingFactorNum, double* d_scalingFactorDenom )
{
    double* reduce_result;

    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (nCells + threads_per_block - 1) / threads_per_block;

    //Purpose: A.Amul get Acf
    AmulGPU(dataBase, d_AcfField, d_Field,
            diag, off_diag_value, csr_row_index_no_diag, csr_col_index_no_diag, 
            interfaceIntCoeffs, interfaceBouCoeffs,
            faceCells, nPatchFaces, nCells);

    double scalingFactor = 0.0;
    double sum_scalingFactorNum = 0.0, sum_scalingFactorDenom = 0.0;

    checkCudaErrors(cudaMemset(d_scalingFactorNum,   0, nCells*sizeof(double)));
    checkCudaErrors(cudaMemset(d_scalingFactorDenom, 0, nCells*sizeof(double)));

    kernel_calcNumDenom<<<blocks_per_grid, threads_per_block, 0, dataBase.stream>>>
        (nCells, d_Field, d_Source, d_AcfField, d_scalingFactorNum, d_scalingFactorDenom);
    checkCudaErrors(cudaStreamSynchronize(dataBase.stream));

    reduce(nCells, threads_per_block, blocks_per_grid, d_scalingFactorNum, reduce_result, dataBase.stream, false);
#ifndef PARALLEL_
    cudaMemcpyAsync(&sum_scalingFactorNum, &reduce_result[0] , sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#else
    ncclAllReduce(&reduce_result[0], &reduce_result[0], 1, ncclDouble, ncclSum, dataBase.nccl_comm, dataBase.stream);
    cudaStreamSynchronize(dataBase.stream);
    cudaMemcpyAsync(&sum_scalingFactorNum, &reduce_result[0], sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#endif

    reduce(nCells, threads_per_block, blocks_per_grid, d_scalingFactorDenom, reduce_result, dataBase.stream, false);
#ifndef PARALLEL_
    cudaMemcpyAsync(&sum_scalingFactorDenom, &reduce_result[0] , sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#else
    ncclAllReduce(&reduce_result[0], &reduce_result[0], 1, ncclDouble, ncclSum, dataBase.nccl_comm, dataBase.stream);
    cudaStreamSynchronize(dataBase.stream);
    cudaMemcpyAsync(&sum_scalingFactorDenom, &reduce_result[0], sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#endif

    std::vector scalingVector(sum_scalingFactorNum, sum_scalingFactorDenom);
    scalingFactor = scalingVector[0]/(scalingVector[1] + 1e-20); // need test stabilise

    kernel_scale<<<blocks_per_grid, threads_per_block, 0, dataBase.stream>>>
        (nCells, scalingFactor, d_Field, d_Source, d_AcfField, diag);
    checkCudaErrors(cudaStreamSynchronize(dataBase.stream));

}

void updateSourceFieldGPU(const dfMatrixDataBase& dataBase, int nCells, 
                        double* d_Sources, double* d_AcfField, double* d_CorrFields,
                        double* diag, double* off_diag_value,
                        int* csr_row_index_no_diag, int* csr_col_index_no_diag, 
                        double** interfaceIntCoeffs, double** interfaceBouCoeffs,
                        int** faceCells, std::vector<int> nPatchFaces)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (nCells + threads_per_block - 1) / threads_per_block;

    //Purpose: spmv to get Acf = A * Corr
    AmulGPU(dataBase, d_AcfField, d_CorrFields,
            diag, off_diag_value, csr_row_index_no_diag, csr_col_index_no_diag, 
            interfaceIntCoeffs, interfaceBouCoeffs, faceCells, nPatchFaces, nCells);

    kernel_updateSource<<<blocks_per_grid, threads_per_block, 0, dataBase.stream>>>
        (nCells, d_Sources, d_AcfField);
    checkCudaErrors(cudaStreamSynchronize(dataBase.stream));
}

void updateCorrFieldGPU(cudaStream_t stream, int nCells, 
                        double* d_Field, double* d_preSmoothField)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (nCells + threads_per_block - 1) / threads_per_block;

    kernel_updateCorr<<<blocks_per_grid, threads_per_block, 0, stream>>>
        (nCells, d_Field, d_preSmoothField);
    checkCudaErrors(cudaStreamSynchronize(stream));
}

void directSolve1x1GPU(cudaStream_t stream, int nCells, 
                        double* d_diag, double* d_corrField, double* d_sourceField)
{
    kernel_directSolve1x1<<<1, 1, 0, stream>>>
        (nCells, d_diag, d_corrField, d_sourceField);
    checkCudaErrors(cudaStreamSynchronize(stream));
}

__global__ void kernel_addInternalInterfaceCoeffs
(
    int num, 
    const double* interfaceIntCoeffs, 
    int* face2Cells, 
    double* diag
)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= num)
        return;

    int cellIndex = face2Cells[index];
    diag[cellIndex] += interfaceIntCoeffs[index];
}

__global__ void kernel_initMatrixInterfacesCoeffs
(
    double* input, 
    int interfaceiSize,
    double* scalarSendBufList_,
    int* face2Cells
)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= interfaceiSize)
        return;

    int cellIndex = face2Cells[index];
    scalarSendBufList_[index] = input[cellIndex];
}

__global__ void kernel_updateMatrixInterfacesCoeffs
(
    const double* interfaceBouCoeffs,
    int interfaceiSize,
    double* scalarRecvBufList_,
    int* face2Cells,
    double* output
)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= interfaceiSize)
        return;

    int cellIndex = face2Cells[index];
    output[cellIndex] -= interfaceBouCoeffs[index] * scalarRecvBufList_[index];
}

void AmulGPU(const dfMatrixDataBase& dataBase, double* result, double* input,
            double* diag, double* off_diag_value,
            int* csr_row_index_no_diag, int* csr_col_index_no_diag, 
            double** interfaceIntCoeffs, double** interfaceBouCoeffs,
            int** faceCells, std::vector<int> nPatchFaces, int nCells)
{
    // --- addInternalInterfaceCoeffs ---
    addInternalInterfaceCoeffs(dataBase.stream, nPatchFaces, interfaceIntCoeffs, faceCells, diag);

    // --- SpMV ---
    SpMV4CSR(dataBase.stream, nCells, diag, off_diag_value, csr_row_index_no_diag, csr_col_index_no_diag, input, result); 

#ifdef PARALLEL_      
    // --- initMatrixInterfaces & updateMatrixInterfaces ---
    updateMatrixInterfaceCoeffs(
        dataBase.stream, dataBase.neighbProcNo, dataBase.nccl_comm,
        nPatchFaces, input, result, 
        scalarSendBufList_, scalarRecvBufList_,
        interfaceBouCoeffs, faceCells);
#endif
}

void addInternalInterfaceCoeffs(
        cudaStream_t stream, std::vector<int> patchSize, 
        double **interfaceIntCoeffs, int **boundaryFaceCell, double *diag) 
{
    for (int patchi = 0; patchi < patchSize.size(); patchi++) 
    {
        size_t threads_per_block = 1024;
        size_t blocks_per_grid = (patchSize[patchi] + threads_per_block - 1) / threads_per_block;

        kernel_addInternalInterfaceCoeffs<<<blocks_per_grid, threads_per_block, 0, stream>>>
            (patchSize[patchi], interfaceIntCoeffs[patchi], boundaryFaceCell[patchi], diag);
    }
}

void updateMatrixInterfaceCoeffs(
    cudaStream_t stream, std::vector<int> neighbProcNo,  ncclComm_t nccl_comm,
    std::vector<int> patchSize, double *input, double *output, 
    double *scalarSendBufList_, double *scalarRecvBufList_,
    double **interfaceBouCoeffs, int **boundaryFaceCell)
{
    for (int patchi = 0; patchi < patchSize.size(); patchi++) 
    {
        size_t threads_per_block = 1024;
        size_t blocks_per_grid = (patchSize[patchi] + threads_per_block - 1) / threads_per_block;

        kernel_initMatrixInterfacesCoeffs<<<blocks_per_grid, threads_per_block, 0, stream>>>
            (input, patchSize[patchi], scalarSendBufList_, boundaryFaceCell[patchi]);

        ncclGroupStart();
        ncclSend(scalarSendBufList_, patchSize[patchi], ncclDouble, neighbProcNo[patchi], nccl_comm, stream);
        ncclRecv(scalarRecvBufList_, patchSize[patchi], ncclDouble, neighbProcNo[patchi], nccl_comm, stream);
        ncclGroupEnd();

        kernel_updateMatrixInterfacesCoeffs<<<blocks_per_grid, threads_per_block, 0, stream>>>
            (interfaceBouCoeffs[patchi], patchSize[patchi], scalarRecvBufList_, boundaryFaceCell[patchi], output);
    }
}

__global__ void kernel_interpolate
(
    int nCells,
    double* Apsi, double* diag, double* psi
)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nCells)
        return;
    
    psi[index] = -Apsi[index]/(diag[index]);
}

__global__ void kernel_interpolateUpdateCoarseLevel
(
    int nCells, int* restrictMap,
    double* corrC, double* diagC, double* diag, double* psi
)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nCells)
        return;

    int mapIndex = restrictMap[index];
    atomicAdd(&corrC[mapIndex], (diag[index] * psi[index]));
    atomicAdd(&diagC[mapIndex], diag[index]);
}

__global__ void kernel_interpolateCoarseCorr
(
    int nCCells, double* corrC, double* diagC, double* psiC
)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nCCells)
        return;

    corrC[index] = psiC[index] - corrC[index] / diagC[index];
}

__global__ void kernel_interpolateFineLevel
(
    int nCells, int* restrictMap,
    double* psi, double* corrC
)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= nCells)
        return;

    int mapIndex = restrictMap[index];
    atomicAdd(&psi[index], corrC[mapIndex]);
}

void interpolateFieldGPU(const dfMatrixDataBase& dataBase, int nCells, int nCCells, 
                    double* psi, double* Apsi, 
                    double* diag, double* off_diag_value,
                    int* csr_row_index_no_diag, int* csr_col_index_no_diag,  
                    double** interfaceIntCoeffs, double** interfaceBouCoeffs, 
                    int** faceCells, std::vector<int> nPatchFaces,
                    int* restrictAddressing, double* psiC)
{
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (nCells + threads_per_block - 1) / threads_per_block;

    // Purpose: m.Amul(Apsi, psi, interfaceBouCoeffs, interfaces, cmpt);
    AmulGPU(dataBase, Apsi, psi,
            diag, off_diag_value, csr_row_index_no_diag, csr_col_index_no_diag, 
            interfaceIntCoeffs, interfaceBouCoeffs,
            faceCells, nPatchFaces, nCells);

    kernel_interpolate<<<blocks_per_grid, threads_per_block, 0, dataBase.stream>>>
        (nCells, Apsi, diag, psi);
    checkCudaErrors(cudaStreamSynchronize(dataBase.stream));

    // tmp data for interpolate can remove to init
    double* corrC;
    double* diagC;

    checkCudaErrors(cudaMalloc(&corrC, nCCells*sizeof(double)));
    checkCudaErrors(cudaMalloc(&diagC, nCCells*sizeof(double)));

    checkCudaErrors(cudaMemset(corrC, 0, nCCells*sizeof(double)));
    checkCudaErrors(cudaMemset(diagC, 0, nCCells*sizeof(double)));

    kernel_interpolateUpdateCoarseLevel<<<blocks_per_grid, threads_per_block, 0, dataBase.stream>>>
        (nCells, restrictAddressing, corrC, diagC, diag, psi);
    checkCudaErrors(cudaStreamSynchronize(dataBase.stream));

    kernel_interpolateCoarseCorr<<<blocks_per_grid, threads_per_block, 0, dataBase.stream>>>
        (nCCells, corrC, diagC, psiC);
    checkCudaErrors(cudaStreamSynchronize(dataBase.stream));

    kernel_interpolateFineLevel<<<blocks_per_grid, threads_per_block, 0, dataBase.stream>>>
        (nCells, restrictAddressing, psi, corrC);
    checkCudaErrors(cudaStreamSynchronize(dataBase.stream));

    checkCudaErrors(cudaFreeAsync(corrC, dataBase.stream));
    checkCudaErrors(cudaFreeAsync(diagC, dataBase.stream));
}