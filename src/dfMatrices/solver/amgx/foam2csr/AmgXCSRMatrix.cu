/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <AmgXCSRMatrix.H>

#include <cuda.h>
#include <cub/cub.cuh>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <numeric>

#include <mpi.h>

#define CHECK(call)                                              \
    {                                                            \
        cudaError_t e = call;                                    \
        if (e != cudaSuccess)                                    \
        {                                                        \
            printf("Cuda failure: '%s %d %s'",                   \
                __FILE__, __LINE__, cudaGetErrorString(e));      \
        }                                                        \
    }

// Offset the column indices to transform from local to global
__global__ void fixConsolidatedRowIndices(
    const int nInternalFaces,
    const int nifDisp,
    const int nExtNz,
    const int extDisp,
    const int nConsRows,
    const int nConsInternalFaces,
    const int offset,
    int *rowIndices)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < nInternalFaces)
    {
        rowIndices[nConsRows + nifDisp + i] += offset;
        rowIndices[nConsRows + nConsInternalFaces + nifDisp + i] += offset;
    }

    if (i < nExtNz)
    {
        rowIndices[nConsRows + 2 * nConsInternalFaces + extDisp + i] += offset;
    }
}

// Offset the column indices to transform from local to global
__global__ void localToGlobalColIndices(
    const int nnz,
    const int nLocalRows,
    const int nInternalFaces,
    const int rowDisp,
    const int nifDisp,
    const int nConsRows,
    const int nConsInternalFaces,
    const int diagIndexGlobal,
    const int lowOffGlobal,
    const int uppOffGlobal,
    int *colIndicesGlobal)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < nLocalRows)
    {
        colIndicesGlobal[rowDisp + i] += diagIndexGlobal;
    }

    if (i < nInternalFaces)
    {
        colIndicesGlobal[nConsRows + nifDisp + i] += uppOffGlobal;
        colIndicesGlobal[nConsRows + nConsInternalFaces + nifDisp + i] += lowOffGlobal;
    }
}

// Apply the pre-existing permutation to the values [and columns]
__global__ void applyPermutation(
    const int nTotalNz,
    const int *perm,
    const int *colIndicesTmp,
    const double *valuesTmp,
    int *colIndicesGlobal,
    double *values,
    bool valuesOnly)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= nTotalNz) return;

    int p = perm[i];

    // In the values only case the column indices and row offsets remain fixed so no update
    if (!valuesOnly)
    {
        colIndicesGlobal[i] = colIndicesTmp[p];
    }

    values[i] = valuesTmp[p];
}

// Flatten the row indices into the row offsets
__global__ void createRowOffsets(
    int nnz,
    int *rowIndices,
    int *rowOffsets)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < nnz; i += blockDim.x * gridDim.x)
    {
        atomicAdd(&rowOffsets[rowIndices[i]], 1);
    }
}

// Convert a float array into a double array
__global__ void floatToDoubleArray(
    int size,
    float *src,
    double *dest)
{
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += blockDim.x * gridDim.x)
    {
        dest[i] = (double)src[i];
    }
}

void AmgXCSRMatrix::initialiseComms(
    MPI_Comm devWorld,
    int gpuProc)
{
    this->devWorld = devWorld;
    this->gpuProc = gpuProc;

    MPI_Comm_rank(this->devWorld, &myDevWorldRank);
    MPI_Comm_size(this->devWorld, &devWorldSize);
}

// Initialises the consolidation feature
void AmgXCSRMatrix::initialiseConsolidation(
    const int nLocalRows,
    const int nLocalNz,
    const int nInternalFaces,
    const int nExtNz,
    int*& rowIndicesTmp,
    int*& colIndicesTmp)
{
    // Consolidation has been previously used, must deallocate the structures
    if (consolidationStatus != ConsolidationStatus::Uninitialised)
    {
        finaliseConsolidation();
    }

    // Check if only one rank is associated with the device of devWorld, and exit
    // early since no consolidation is done in this case:
    if (devWorldSize == 1)
    {
        // This value will be the same for all ranks within devWorld
        consolidationStatus = ConsolidationStatus::None;

        // Allocate data only
        CHECK(cudaMalloc((void **)&rowIndicesTmp, (nLocalNz + nExtNz) * sizeof(int)));
        CHECK(cudaMalloc((void **)&colIndicesTmp, (nLocalNz + nExtNz) * sizeof(int)));
        CHECK(cudaMalloc((void **)&valuesTmp, (nLocalNz + nExtNz) * sizeof(double)));
        CHECK(cudaMalloc((void **)&fvaluesTmp, (nLocalNz + nExtNz) * sizeof(float)));
        return;
    }

    nRowsInDevWorld.resize(devWorldSize);
    nnzInDevWorld.resize(devWorldSize);
    nInternalFacesInDevWorld.resize(devWorldSize);
    nExtNzInDevWorld.resize(devWorldSize);

    rowDispls.resize(devWorldSize + 1, 0);
    nzDispls.resize(devWorldSize + 1, 0);
    internalFacesDispls.resize(devWorldSize + 1, 0);
    extNzDispls.resize(devWorldSize + 1, 0);

    // Fetch to all the number of local rows and non zeros on each rank
    MPI_Request reqs[4] = { MPI_REQUEST_NULL};
    MPI_Iallgather(&nLocalRows, 1, MPI_INT, nRowsInDevWorld.data(), 1, MPI_INT, devWorld, &reqs[0]);
    MPI_Iallgather(&nLocalNz, 1, MPI_INT, nnzInDevWorld.data(), 1, MPI_INT, devWorld, &reqs[1]);
    MPI_Iallgather(&nInternalFaces, 1, MPI_INT, nInternalFacesInDevWorld.data(), 1, MPI_INT, devWorld, &reqs[2]);
    MPI_Iallgather(&nExtNz, 1, MPI_INT, nExtNzInDevWorld.data(), 1, MPI_INT, devWorld, &reqs[3]);
    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

    // Calculate consolidate number of rows, non-zeros, and calculate row, non-zero displacements
    std::partial_sum(nRowsInDevWorld.begin(), nRowsInDevWorld.end(), rowDispls.begin() + 1);
    std::partial_sum(nnzInDevWorld.begin(), nnzInDevWorld.end(), nzDispls.begin() + 1);
    std::partial_sum(nInternalFacesInDevWorld.begin(), nInternalFacesInDevWorld.end(), internalFacesDispls.begin() + 1);
    std::partial_sum(nExtNzInDevWorld.begin(), nExtNzInDevWorld.end(), extNzDispls.begin() + 1);

    nConsNz = nzDispls[devWorldSize];
    nConsRows = rowDispls[devWorldSize];
    nConsInternalFaces = internalFacesDispls[devWorldSize];
    nConsExtNz = extNzDispls[devWorldSize];

    // Consolidate the CSR matrix data from multiple ranks sharing a single GPU to a
    // root rank, allowing multiple ranks per GPU. This allows overdecomposing the problem
    // when there are more CPU cores than GPUs, without the inefficiences of performing
    // the linear solve on multiple separate domains.
    ConsolidationHandles handles;

    // The data is already on the GPU so consolidate there
    if (gpuProc == 0)
    {
        // We are consolidating data that already exists on the GPU
        CHECK(cudaMalloc((void **)&rhsCons, sizeof(double) * nConsRows));
        CHECK(cudaMalloc((void **)&pCons, sizeof(double) * nConsRows));
        CHECK(cudaMalloc((void **)&rowIndicesTmp, sizeof(int) * (nConsNz + nConsExtNz)));
        CHECK(cudaMalloc((void **)&colIndicesTmp, sizeof(int) * (nConsNz + nConsExtNz)));
        CHECK(cudaMalloc((void **)&valuesTmp, sizeof(double) * (nConsNz + nConsExtNz)));
        CHECK(cudaMalloc((void **)&fvaluesTmp, sizeof(float) * (nConsNz + nConsExtNz)));

        CHECK(cudaIpcGetMemHandle(&handles.rhsConsHandle, rhsCons));
        CHECK(cudaIpcGetMemHandle(&handles.solConsHandle, pCons));
        CHECK(cudaIpcGetMemHandle(&handles.rowIndicesConsHandle, rowIndicesTmp));
        CHECK(cudaIpcGetMemHandle(&handles.colIndicesConsHandle, colIndicesTmp));
        CHECK(cudaIpcGetMemHandle(&handles.valuesConsHandle, valuesTmp));
        CHECK(cudaIpcGetMemHandle(&handles.fvaluesConsHandle, fvaluesTmp));
    }

    MPI_Bcast(&handles, sizeof(ConsolidationHandles), MPI_BYTE, 0, devWorld);

    // Open memory handles to the consolidated matrix data owned by the gpu owning process
    if (gpuProc == MPI_UNDEFINED)
    {
        CHECK(cudaIpcOpenMemHandle((void **)&rhsCons, handles.rhsConsHandle, cudaIpcMemLazyEnablePeerAccess));
        CHECK(cudaIpcOpenMemHandle((void **)&pCons, handles.solConsHandle, cudaIpcMemLazyEnablePeerAccess));
        CHECK(cudaIpcOpenMemHandle((void **)&rowIndicesTmp, handles.rowIndicesConsHandle, cudaIpcMemLazyEnablePeerAccess));
        CHECK(cudaIpcOpenMemHandle((void **)&colIndicesTmp, handles.colIndicesConsHandle, cudaIpcMemLazyEnablePeerAccess));
        CHECK(cudaIpcOpenMemHandle((void **)&valuesTmp, handles.valuesConsHandle, cudaIpcMemLazyEnablePeerAccess));
        CHECK(cudaIpcOpenMemHandle((void **)&fvaluesTmp, handles.fvaluesConsHandle, cudaIpcMemLazyEnablePeerAccess));
    }

    // This value will be the same for all ranks within devWorld
    consolidationStatus = ConsolidationStatus::Device;
}

// Perform the conversion between an LDU matrix and a CSR matrix, possibly distributed
void AmgXCSRMatrix::setValuesLDU
(
    int nLocalRows,
    int nInternalFaces,
    int diagIndexGlobal,
    int lowOffGlobal,
    int uppOffGlobal,
    const int *upperAddr,
    const int *lowerAddr,
    const int nExtNz,
    const int *extRow,
    const int *extCol,
    const float *diagVals,
    const float *upperVals,
    const float *lowerVals,
    const float *extVals
)
{
    // Make a copy of the host vectors, converting all floats to doubles
    double* ddiagVals  = new double[nLocalRows];
    double* dupperVals = new double[nInternalFaces];
    double* dlowerVals = new double[nInternalFaces];
    double* dextVals   = new double[nExtNz];

    for (int i=0; i<nLocalRows; ++i)
    {
	ddiagVals[i] = (double)diagVals[i];
    }

    for (int i=0; i<nInternalFaces; ++i)
    {
        dupperVals[i] = (double)upperVals[i];
	dlowerVals[i] = (double)lowerVals[i];
    }

    for (int i=0; i<nExtNz; ++i)
    {
        dextVals[i] = (double)extVals[i];
    }

    setValuesLDU
    (
        nLocalRows,
        nInternalFaces,
        diagIndexGlobal,
        lowOffGlobal,
        uppOffGlobal,
        upperAddr,
        lowerAddr,
        nExtNz,
        extRow,
        extCol,
        ddiagVals,
        dupperVals,
        dlowerVals,
        dextVals
    );

    delete[] ddiagVals;
    delete[] dupperVals;
    delete[] dlowerVals;
    delete[] dextVals;

    return;
}

// Perform the conversion between an LDU matrix and a CSR matrix, possibly distributed
void AmgXCSRMatrix::setValuesLDU
(
    int nLocalRows,
    int nInternalFaces,
    int diagIndexGlobal,
    int lowOffGlobal,
    int uppOffGlobal,
    const int *upperAddr,
    const int *lowerAddr,
    const int nExtNz,
    const int *extRow,
    const int *extCol,
    const double *diagVals,
    const double *upperVals,
    const double *lowerVals,
    const double *extVals
)
{
    // Determine the local non-zeros from the internal faces
    int nLocalNz = nLocalRows + 2 * nInternalFaces;
    int *rowIndicesTmp;
    int *permTmp;
    int *colIndicesTmp;

    initialiseConsolidation(nLocalRows, nLocalNz, nInternalFaces, nExtNz, rowIndicesTmp, colIndicesTmp);

    int nTotalNz = 0;
    int nRows = 0;

    std::vector<int> diagIndexGlobalAll(devWorldSize);
    std::vector<int> lowOffGlobalAll(devWorldSize);
    std::vector<int> uppOffGlobalAll(devWorldSize);

    switch (consolidationStatus)
    {

    case ConsolidationStatus::None:
    {
        nTotalNz = nLocalNz + nExtNz;
        nRows = nLocalRows;

        // Generate unpermuted index list [0, ..., nTotalNz-1]
        CHECK(cudaMalloc(&permTmp, sizeof(int) * nTotalNz));
        thrust::sequence(thrust::device, permTmp, permTmp + nTotalNz, 0);

        // Fill rowIndicesTmp with [0, ..., n-1], lowerAddr, upperAddr, (extAddr)
        thrust::sequence(thrust::device, rowIndicesTmp, rowIndicesTmp + nRows, 0);
        CHECK(cudaMemcpy(rowIndicesTmp + nRows, lowerAddr, nInternalFaces * sizeof(int), cudaMemcpyDefault));
        CHECK(cudaMemcpy(rowIndicesTmp + nRows + nInternalFaces, upperAddr, nInternalFaces * sizeof(int), cudaMemcpyDefault));
        if (nExtNz > 0)
        {
            CHECK(cudaMemcpy(rowIndicesTmp + nLocalNz, extRow, nExtNz * sizeof(int), cudaMemcpyDefault));
        }

        // Concat [0, ..., n-1], upperAddr, lowerAddr (note switched) into column indices
        thrust::sequence(thrust::device, colIndicesTmp, colIndicesTmp + nRows, 0);
        CHECK(cudaMemcpy(colIndicesTmp + nRows, rowIndicesTmp + nRows + nInternalFaces, nInternalFaces * sizeof(int), cudaMemcpyDefault));
        CHECK(cudaMemcpy(colIndicesTmp + nRows + nInternalFaces, rowIndicesTmp + nRows, nInternalFaces * sizeof(int), cudaMemcpyDefault));
        if (nExtNz > 0)
        {
            CHECK(cudaMemcpy(colIndicesTmp + nLocalNz, extCol, nExtNz * sizeof(int), cudaMemcpyDefault));
        }

        // Fill valuesTmp with diagVals, upperVals, lowerVals, (extVals)
        CHECK(cudaMemcpy(valuesTmp, diagVals, nRows * sizeof(double), cudaMemcpyDefault));
        CHECK(cudaMemcpy(valuesTmp + nRows, upperVals, nInternalFaces * sizeof(double), cudaMemcpyDefault));
        CHECK(cudaMemcpy(valuesTmp + nRows + nInternalFaces, lowerVals, nInternalFaces * sizeof(double), cudaMemcpyDefault));
        if (nExtNz > 0)
        {
            CHECK(cudaMemcpy(valuesTmp + nLocalNz, extVals, nExtNz * sizeof(double), cudaMemcpyDefault));
        }
        break;
    }
    case ConsolidationStatus::Device:
    {
        nTotalNz = nConsNz + nConsExtNz;
        nRows = nConsRows;

        // Copy the data to the consolidation buffer
        // Fill rowIndicesTmp with lowerAddr, upperAddr, (extAddr)
        CHECK(cudaMemcpy(rowIndicesTmp + nConsRows + internalFacesDispls[myDevWorldRank], lowerAddr, nInternalFaces * sizeof(int), cudaMemcpyDefault));
        CHECK(cudaMemcpy(rowIndicesTmp + nConsRows + nConsInternalFaces + internalFacesDispls[myDevWorldRank], upperAddr, nInternalFaces * sizeof(int), cudaMemcpyDefault));

        // Fill valuesTmp with diagVals, upperVals, lowerVals, (extVals)
        CHECK(cudaMemcpy(valuesTmp + rowDispls[myDevWorldRank], diagVals, nLocalRows * sizeof(double), cudaMemcpyDefault));
        CHECK(cudaMemcpy(valuesTmp + nConsRows + internalFacesDispls[myDevWorldRank], upperVals, nInternalFaces * sizeof(double), cudaMemcpyDefault));
        CHECK(cudaMemcpy(valuesTmp + nConsRows + nConsInternalFaces + internalFacesDispls[myDevWorldRank], lowerVals, nInternalFaces * sizeof(double), cudaMemcpyDefault));
        if (nExtNz > 0)
        {
            CHECK(cudaMemcpy(rowIndicesTmp + nConsNz + extNzDispls[myDevWorldRank], extRow, nExtNz * sizeof(int), cudaMemcpyDefault));
            CHECK(cudaMemcpy(colIndicesTmp + nConsNz + extNzDispls[myDevWorldRank], extCol, nExtNz * sizeof(int), cudaMemcpyDefault));
            CHECK(cudaMemcpy(valuesTmp + nConsNz + extNzDispls[myDevWorldRank], extVals, nExtNz * sizeof(double), cudaMemcpyDefault));
        }

        // cudaMemcpy does not block the host in the cases above, device to device copies,
        // so sychronize with device to ensure operation is complete. Barrier on all devWorld
        // ranks to ensure full arrays are populated before the root process uses the data.
        CHECK(cudaDeviceSynchronize());
        int ierr = MPI_Barrier(devWorld);

        if (gpuProc == 0)
        {
            // Generate unpermuted index list [0, ..., nTotalNz]
            CHECK(cudaMalloc(&permTmp, sizeof(int) * nTotalNz));
            thrust::sequence(thrust::device, permTmp, permTmp + nTotalNz, 0);

            // Concat [0, ..., n-1], upperAddr, lowerAddr (note switched) into column indices
            thrust::sequence(thrust::device, rowIndicesTmp, rowIndicesTmp + nConsRows, 0);
            CHECK(cudaMemcpy(colIndicesTmp + nConsRows, rowIndicesTmp + nConsRows + nConsInternalFaces, nConsInternalFaces * sizeof(int), cudaMemcpyDefault));
            CHECK(cudaMemcpy(colIndicesTmp + nConsRows + nConsInternalFaces, rowIndicesTmp + nConsRows, nConsInternalFaces * sizeof(int), cudaMemcpyDefault));

            for (int i = 0; i < devWorldSize; ++i)
            {
                int nrows = nRowsInDevWorld[i];
                int rowDisp = rowDispls[i];

                thrust::sequence(thrust::device, colIndicesTmp + rowDisp, colIndicesTmp + rowDisp + nrows, 0);

                // Skip first as offset 0
                if (i > 0)
                {
                    int nif = nInternalFacesInDevWorld[i];
                    int nifDisp = internalFacesDispls[i];
                    int extDisp = extNzDispls[i];
                    int nenz = nExtNzInDevWorld[i];

                    // Adjust rowIndices so that they are correct for the consolidated matrix
                    int nthreads = 128;
                    int nblocks = nif / nthreads + 1;
                    fixConsolidatedRowIndices<<<nblocks, nthreads>>>(nif, nifDisp, nenz, extDisp, nConsRows, nConsInternalFaces, rowDisp, rowIndicesTmp);
                }
            }
        }
        else
        {
            // Close IPC handles and deallocate for consolidation
            CHECK(cudaIpcCloseMemHandle(rowIndicesTmp));
            CHECK(cudaIpcCloseMemHandle(colIndicesTmp));
        }

        MPI_Request reqs[3] = { MPI_REQUEST_NULL };
        MPI_Igather(&diagIndexGlobal, 1, MPI_INT, diagIndexGlobalAll.data(), 1, MPI_INT, 0, devWorld, &reqs[0]);
        MPI_Igather(&lowOffGlobal, 1, MPI_INT, lowOffGlobalAll.data(), 1, MPI_INT, 0, devWorld, &reqs[1]);
        MPI_Igather(&uppOffGlobal, 1, MPI_INT, uppOffGlobalAll.data(), 1, MPI_INT, 0, devWorld, &reqs[2]);
        MPI_Waitall(3, reqs, MPI_STATUSES_IGNORE);

        break;
    }
    case ConsolidationStatus::Uninitialised:
    {
        fprintf(stderr, "Attempting to consolidate before consolidation is initialised.\n");
        break;
    }
    default:
    {
        fprintf(stderr, "Incorrect consolidation status set.\n");
        break;
    }
    }

    if (gpuProc == 0)
    {
        // Make space for the row indices and stored permutation
        int *rowIndices;
        CHECK(cudaMalloc(&rowIndices, sizeof(int) * nTotalNz));
        CHECK(cudaMalloc(&ldu2csrPerm, sizeof(int) * nTotalNz));

        cub::DoubleBuffer<int> d_keys(rowIndicesTmp, rowIndices);
        cub::DoubleBuffer<int> d_values(permTmp, ldu2csrPerm);

        // Sort the row indices and store results in the permutation
        void *tempStorage = NULL;
        size_t tempStorageBytes = 0;
        cub::DeviceRadixSort::SortPairs(tempStorage, tempStorageBytes, d_keys, d_values, nTotalNz);
        CHECK(cudaMalloc(&tempStorage, tempStorageBytes));
        cub::DeviceRadixSort::SortPairs(tempStorage, tempStorageBytes, d_keys, d_values, nTotalNz);
        if(tempStorageBytes > 0)
        {
            CHECK(cudaFree(tempStorage));
        }

        // Fetch the invalid pointers from the CUB ping pong buffers and de-alloc
        CHECK(cudaFree(d_keys.Alternate()));
        CHECK(cudaFree(d_values.Alternate()));

        // Fetch the correct pointers from the CUB ping pong buffers
        rowIndices = d_keys.Current();
        ldu2csrPerm = d_values.Current();

        // Make space for the row offsets
        CHECK(cudaMalloc(&rowOffsets, sizeof(int) * (nRows + 1)));
        CHECK(cudaMemset(rowOffsets, 0, sizeof(int) * (nRows + 1)));

        // Convert the row indices into offsets
        constexpr int nthreads = 128;
        int nblocks = nTotalNz / nthreads + 1;
        createRowOffsets<<<nblocks, nthreads>>>(nTotalNz, rowIndices, rowOffsets);
        thrust::exclusive_scan(thrust::device, rowOffsets, rowOffsets + nRows + 1, rowOffsets);
        CHECK(cudaFree(rowIndices));

        // Transform the local column indices to global column indices
        if (isConsolidated())
        {
            for (int i = 0; i < devWorldSize; ++i)
            {
                nblocks = nnzInDevWorld[i] / nthreads + 1;
                localToGlobalColIndices<<<nblocks, nthreads>>>(nnzInDevWorld[i], nRowsInDevWorld[i], nInternalFacesInDevWorld[i],
                                                               rowDispls[i], internalFacesDispls[i], nConsRows,
                                                               nConsInternalFaces, diagIndexGlobalAll[i],
                                                               lowOffGlobalAll[i], uppOffGlobalAll[i], colIndicesTmp);
            }
        }
        else
        {
            nblocks = nLocalNz / nthreads + 1;
            localToGlobalColIndices<<<nblocks, nthreads>>>(nLocalNz, nLocalRows, nInternalFaces, 0, 0,
                                                           nLocalRows, nInternalFaces, diagIndexGlobal,
                                                           lowOffGlobal, uppOffGlobal, colIndicesTmp);
        }

        // Allocate space to store the permuted column indices and values
        CHECK(cudaMalloc(&colIndicesGlobal, sizeof(int) * nTotalNz));
        CHECK(cudaMalloc(&values, sizeof(double) * nTotalNz));

        // Swap column indices based on the pre-determined permutation
        nblocks = nTotalNz / nthreads + 1;
        applyPermutation<<<nblocks, nthreads>>>(nTotalNz, ldu2csrPerm, colIndicesTmp, valuesTmp, colIndicesGlobal, values, false);

        CHECK(cudaFree(colIndicesTmp));
    }
}

// Updates the values based on the previously determined permutation
void AmgXCSRMatrix::updateValues
(
    const int nLocalRows,
    const int nInternalFaces,
    const int nExtNz,
    const float *diagVals,
    const float *upperVals,
    const float *lowerVals,
    const float *extVals
)
{
// Add external non-zeros (communicated halo entries)
    int nTotalNz;

    if (isConsolidated())
    {
        nTotalNz = (nConsNz + nConsExtNz);

        constexpr int nthreads = 128;
        int nblocks = nTotalNz / nthreads + 1;

        // Fill (f)valuesTmp with diagVals, upperVals, lowerVals, (extVals)
        CHECK(cudaMemcpy(fvaluesTmp + rowDispls[myDevWorldRank], diagVals, nLocalRows * sizeof(float), cudaMemcpyDefault));        
        CHECK(cudaMemcpy(fvaluesTmp + nConsRows + internalFacesDispls[myDevWorldRank], upperVals, nInternalFaces * sizeof(float), cudaMemcpyDefault));
        CHECK(cudaMemcpy(fvaluesTmp + nConsRows + nConsInternalFaces + internalFacesDispls[myDevWorldRank], lowerVals, nInternalFaces * sizeof(float), cudaMemcpyDefault));

        if (nExtNz > 0)
        {
            CHECK(cudaMemcpy(fvaluesTmp + nConsNz + extNzDispls[myDevWorldRank], extVals, nExtNz * sizeof(float), cudaMemcpyDefault));
        }

        // convert all float arrays into double arrays
        floatToDoubleArray<<<nblocks, nthreads>>>(nTotalNz, fvaluesTmp, valuesTmp);

        // Ensure that all ranks associated with a device have completed prior to the subsequent permutation
        CHECK(cudaDeviceSynchronize());
        MPI_Barrier(devWorld);
    }
    else
    {
        int nLocalNz = nLocalRows + 2 * nInternalFaces;
        nTotalNz = nLocalNz + nExtNz;

        constexpr int nthreads = 128;
        int nblocks = nTotalNz / nthreads + 1;

        // Copy the values in [ diag, upper, lower, (external) ]
        CHECK(cudaMemcpy(fvaluesTmp, diagVals, nLocalRows * sizeof(float), cudaMemcpyDefault));
        CHECK(cudaMemcpy(fvaluesTmp + nLocalRows, upperVals, nInternalFaces * sizeof(float), cudaMemcpyDefault));
        CHECK(cudaMemcpy(fvaluesTmp + nLocalRows + nInternalFaces, lowerVals, nInternalFaces * sizeof(float), cudaMemcpyDefault));
        if (nExtNz > 0)
        {
            CHECK(cudaMemcpy(fvaluesTmp + nLocalNz, extVals, nExtNz * sizeof(float), cudaMemcpyDefault));
        }

        floatToDoubleArray<<<nblocks, nthreads>>>(nTotalNz, fvaluesTmp, valuesTmp);
    }

    if (gpuProc == 0)
    {
        constexpr int nthreads = 128;
        int nblocks = nTotalNz / nthreads + 1;
        applyPermutation<<<nblocks, nthreads>>>(nTotalNz, ldu2csrPerm, nullptr, valuesTmp, nullptr, values, true);

        // Sync to ensure API errors are caught within the API code and avoid any 
        // issues if users are subsequently using non-blocking streams.
        CHECK(cudaDeviceSynchronize());
    }

    return;
}

// Updates the values based on the previously determined permutation
void AmgXCSRMatrix::updateValues
(
    const int nLocalRows,
    const int nInternalFaces,
    const int nExtNz,
    const double *diagVals,
    const double *upperVals,
    const double *lowerVals,
    const double *extVals
)
{
    // Add external non-zeros (communicated halo entries)
    int nTotalNz;

    if (isConsolidated())
    {
        nTotalNz = (nConsNz + nConsExtNz);

        // Fill valuesTmp with diagVals, upperVals, lowerVals, (extVals)
        CHECK(cudaMemcpy(valuesTmp + rowDispls[myDevWorldRank], diagVals, nLocalRows * sizeof(double), cudaMemcpyDefault));
        CHECK(cudaMemcpy(valuesTmp + nConsRows + internalFacesDispls[myDevWorldRank], upperVals, nInternalFaces * sizeof(double), cudaMemcpyDefault));
        CHECK(cudaMemcpy(valuesTmp + nConsRows + nConsInternalFaces + internalFacesDispls[myDevWorldRank], lowerVals, nInternalFaces * sizeof(double), cudaMemcpyDefault));

        if (nExtNz > 0)
        {
            CHECK(cudaMemcpy(valuesTmp + nConsNz + extNzDispls[myDevWorldRank], extVals, nExtNz * sizeof(double), cudaMemcpyDefault));
        }

        // Ensure that all ranks associated with a device have completed prior to the subsequent permutation
        CHECK(cudaDeviceSynchronize());
        MPI_Barrier(devWorld);
    }
    else
    {
        int nLocalNz = nLocalRows + 2 * nInternalFaces;
        nTotalNz = nLocalNz + nExtNz;

        // Copy the values in [ diag, upper, lower, (external) ]
        CHECK(cudaMemcpy(valuesTmp, diagVals, sizeof(double) * nLocalRows, cudaMemcpyDefault));
        CHECK(cudaMemcpy(valuesTmp + nLocalRows, upperVals, sizeof(double) * nInternalFaces, cudaMemcpyDefault));
        CHECK(cudaMemcpy(valuesTmp + nLocalRows + nInternalFaces, lowerVals, sizeof(double) * nInternalFaces, cudaMemcpyDefault));
        if (nExtNz > 0)
        {
            CHECK(cudaMemcpy(valuesTmp + nLocalNz, extVals, sizeof(double) * nExtNz, cudaMemcpyDefault));
        }
    }

    if (gpuProc == 0)
    {
        constexpr int nthreads = 128;
        int nblocks = nTotalNz / nthreads + 1;
        applyPermutation<<<nblocks, nthreads>>>(nTotalNz, ldu2csrPerm, nullptr, valuesTmp, nullptr, values, true);

        // Sync to ensure API errors are caught within the API code and avoid any 
        // issues if users are subsequently using non-blocking streams.
        CHECK(cudaDeviceSynchronize());
    }
}

// Deallocate remaining storage
void AmgXCSRMatrix::finalise()
{
    switch (consolidationStatus)
    {

    case ConsolidationStatus::None:
    {
        CHECK(cudaFree(ldu2csrPerm));
        CHECK(cudaFree(rowOffsets));
        CHECK(cudaFree(colIndicesGlobal));
        CHECK(cudaFree(values));
        break;
    }
    case ConsolidationStatus::Uninitialised:
    {
        // Consolidation is not required for uninitialised
        return;
    }
    case ConsolidationStatus::Device:
    {
        if (gpuProc == 0)
        {
            // Deallocate the CSR matrix values, solution and RHS
            CHECK(cudaFree(pCons));
            CHECK(cudaFree(rhsCons));
            CHECK(cudaFree(valuesTmp));
            CHECK(cudaFree(fvaluesTmp));
            CHECK(cudaFree(ldu2csrPerm));
            CHECK(cudaFree(rowOffsets));
            CHECK(cudaFree(colIndicesGlobal));
            CHECK(cudaFree(values));
        }
        else
        {
            // Close the remaining IPC memory handles
            CHECK(cudaIpcCloseMemHandle(pCons));
            CHECK(cudaIpcCloseMemHandle(rhsCons));
            CHECK(cudaIpcCloseMemHandle(valuesTmp));
            CHECK(cudaIpcCloseMemHandle(fvaluesTmp));
        }
        break;
    }
    default:
    {
        fprintf(stderr,
                "Incorrect consolidation status set.\n");
        break;
    }
    }

    // Free the local GPU partitioning structures
    if (isConsolidated())
    {
        nRowsInDevWorld.clear();
        nnzInDevWorld.clear();
        rowDispls.clear();
        nzDispls.clear();
    }

    consolidationStatus = ConsolidationStatus::Uninitialised;
}

void AmgXCSRMatrix::finaliseConsolidation()
{
}
