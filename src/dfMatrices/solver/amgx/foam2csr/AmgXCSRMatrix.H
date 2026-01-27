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

#pragma once

#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>

/** \brief A set of handles to the device data storing a consolidated CSR matrix. */
struct ConsolidationHandles
{
    cudaIpcMemHandle_t rhsConsHandle;
    cudaIpcMemHandle_t solConsHandle;
    cudaIpcMemHandle_t rowIndicesConsHandle;
    cudaIpcMemHandle_t colIndicesConsHandle;
    cudaIpcMemHandle_t valuesConsHandle;
    cudaIpcMemHandle_t fvaluesConsHandle;
};

/** \brief Enumeration for the status of matrix consolidation for the solver.*/
enum class ConsolidationStatus
{
    Uninitialised,
    None,
    Device
};

class AmgXCSRMatrix
{
    public:

        // Perform the conversion between an LDU matrix and a CSR matrix,
        // possibly distributed
        void setValuesLDU
        (
            int nrows,
            int nInternalFaces,
            int diagIndexGlobal,
            int lowOffGlobal,
            int uppOffGlobal,
            const int *upperAddr,
            const int *lowerAddr,
            const int extNnz,
            const int *extRow,
            const int *extCol,
            const float *diagVals,
            const float *upperVals,
            const float *lowerVals,
            const float *extVals
        );

        // Perform the conversion between an LDU matrix and a CSR matrix,
        // possibly distributed
        void setValuesLDU
        (
            int nrows,
            int nInternalFaces,
            int diagIndexGlobal,
            int lowOffGlobal,
            int uppOffGlobal,
            const int *upperAddr,
            const int *lowerAddr,
            const int extNnz,
            const int *extRow,
            const int *extCol,
            const double *diagVals,
            const double *upperVals,
            const double *lowerVals,
            const double *extVals
        );

        // Updates the CSR matrix values based on LDU matrix values, permuted with
        // the previously determined permutation
        void updateValues
        (
            const int nrows,
            const int nInternalFaces,
            const int extNnz,
            const float *diagVal,
            const float *upperVal,
            const float *lowerVal,
            const float *extVals
        );

        // Updates the CSR matrix values based on LDU matrix values, permuted with
        // the previously determined permutation
        void updateValues
        (
            const int nrows,
            const int nInternalFaces,
            const int extNnz,
            const double *diagVal,
            const double *upperVal,
            const double *lowerVal,
            const double *extVals
        );

        void initialiseComms(
            MPI_Comm devWorld,
            int gpuProc);

        const int* getColIndices() const
        {
            return colIndicesGlobal;
        }

        const int* getRowOffsets() const
        {
            return rowOffsets;
        }

        const double* getValues() const
        {
            return values;
        }

        double* getPCons()
        {
            return pCons;
        }

        double* getRHSCons()
        {
            return rhsCons;
        }

        const int* getRowDispls() const
        {
            return rowDispls.data();
        }

        int getNConsRows() const
        {
            return nConsRows;
        }

        int getNConsNz() const
        {
            return nConsNz + nConsExtNz;
        }

        bool isConsolidated() const
        {
            return consolidationStatus == ConsolidationStatus::Device;
        }

        // Discard elements of the matrix structure
        void discardStructure();

        // Finalise all data
        void finalise();

    private:

        void initialiseConsolidation(
            const int nLocalRows,
            const int nLocalNz,
            const int nInternalFaces,
            const int nExtNz,
            int*& rowIndicesTmp,
            int*& colIndicesTmp);

        void finaliseConsolidation();

        // CSR device data for AmgX matrix
        int *colIndicesGlobal = nullptr;

        int *rowOffsets = nullptr;

        double *values = nullptr;

        /** \brief (double) Temporary storage for the permutation. */
        double *valuesTmp = nullptr;

        /** \brief (floats) Temporary storage for the permutation. */
        float *fvaluesTmp = nullptr;

        /** \brief The consolidated solution vector. */
        double* pCons = nullptr;

        /** \brief The consolidated right hand side vector. */
        double* rhsCons = nullptr;

        /** \brief A flag indicating the type of consolidation applied, if any.
         * This will be consistent for all ranks within a devWorld. */
        ConsolidationStatus consolidationStatus = ConsolidationStatus::Uninitialised;

        /** \brief Permutation array to convert between LDU and CSR. */
        /** Also possibly encodes sorting of columns. */
        int *ldu2csrPerm = nullptr;

        /** \brief The number of non-zeros consolidated from multiple ranks to a device.*/
        int nConsNz = 0;

        /** \brief The number of rows consolidated from multiple ranks to a device.*/
        int nConsRows = 0;

        /** \brief The number of internal faces consolidation. */
        int nConsInternalFaces = 0;

        /** \brief The number of non zeros consolidated. */
        int nConsExtNz = 0;

        /** \brief The number of rows per rank associated with a single device.*/
        std::vector<int> nRowsInDevWorld {};

        /** \brief The number of non-zeros per rank associated with a single device.*/
        std::vector<int> nnzInDevWorld {};

        /** \brief The number of internal faces per rank associated with a single device.*/
        std::vector<int> nInternalFacesInDevWorld {};

        /** \brief The row displacements per rank associated with a single device.*/
        std::vector<int> rowDispls {};

        /** \brief The internal face count displacements per rank associated with a single device.*/
        std::vector<int> internalFacesDispls {};

        /** \brief The non-zero displacements per rank associated with a single device.*/
        std::vector<int> nzDispls {};

        /** \brief The number of external non zeros per rank associated with a single device.*/
        std::vector<int> nExtNzInDevWorld {};

        /** \brief The external non zero displacements per rank associated with a single device.*/
        std::vector<int> extNzDispls {};

        /** \brief A communicator for processes sharing the same device. */
        MPI_Comm devWorld = nullptr;

        /** \brief A flag indicating if this process will send compute requests to a device. */
        int gpuProc = MPI_UNDEFINED;

        /** \brief Size of \ref AmgXSolver::devWorld "devWorld". */
        int devWorldSize = 0;

        /** \brief Rank in \ref AmgXSolver::devWorld "devWorld". */
        int myDevWorldRank = 0;
};

