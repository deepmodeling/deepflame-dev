/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2018 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Description
    Multiply a given vector (second argument) by the matrix or its transpose
    and return the result in the first argument.

\*---------------------------------------------------------------------------*/

#include "divMatrix.H"
#include <cassert>
#include <omp.h>
#include <mpi.h>
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h> 
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
void Foam::divMatrix::Jacobi
(
    scalarField& psi,
    const scalarField& bPrime
) const
{
    double start = MPI_Wtime();
    if(block_count_ > 1){
        if(row_block_size_ == 32){
#ifdef __ARM_FEATURE_SVE
            Jacobi_split_unroll32_sve(psi, bPrime);
#else
            Jacobi_split(psi, bPrime);
#endif
        }else{
            Jacobi_split(psi, bPrime);
        }
    }else{
        Jacobi_naive(psi, bPrime);
    }
    double end = MPI_Wtime();
    Jacobi_time_ += end - start;

    flops_ += row_ * (distance_count_ + 1) * 2.;
}

void Foam::divMatrix::Jacobi_naive
(
    scalarField& psi,
    const scalarField& bPrime
) const
{
    scalarField psiCopy(row_);

    const scalar* const __restrict__ bPrimePtr = bPrime.begin();

    scalar* __restrict__ psiCopyPtr = psiCopy.begin();
    scalar* __restrict__ psiPtr = psi.begin();

    const scalar* const __restrict__ diagPtr = diag_value_.begin();
    const scalar* const __restrict__ off_diag_value_Ptr = off_diag_value_.begin();

    #pragma omp parallel 
    {
        #pragma omp for
        for(label row = 0; row < row_; ++row){
            psiCopyPtr[row] = psiPtr[row];
        }
        #pragma omp for
        for(label bi = 0; bi < block_count_; ++bi){
            label rbs = DIV_BLOCK_START(bi);
            label rbe = DIV_BLOCK_END(rbs);
            label rbl = DIV_BLOCK_LEN(rbs,rbe);
            scalar* __restrict__ psiPtr_offset = psiPtr + rbs;
            const scalar* const __restrict__ bPrimePtr_offset = bPrimePtr + rbs;
            const scalar* const __restrict__ diagPtr_offset = diagPtr + rbs;
            for(label br = 0; br < rbl; ++br){
                psiPtr_offset[br] = bPrimePtr_offset[br];
            }
            label index_block_start = DIV_INDEX_BLOCK_START(rbs);
            for(label divcol = 0; divcol < distance_count_; ++divcol){
                label index_divcol_start = index_block_start + DIV_COL_OFFSET(divcol);
                label distance = distance_list_[divcol];
                if(distance + rbs >= 0 && distance + rbe <= row_){
                    for(label br = 0; br < rbl; ++br){
                        psiPtr_offset[br] -= off_diag_value_Ptr[index_divcol_start + br] * psiCopyPtr[distance + rbs + br];
                    }
                }else{
                    for(label br = 0; br < rbl; ++br){
                        label col = distance + rbs + br;
                        col = std::max(col, static_cast<label>(0));
                        col = std::min(col, row_ - 1);
                        psiPtr_offset[br] -= off_diag_value_Ptr[index_divcol_start + br] * psiCopyPtr[col];
                    }
                }
            }
            for(label br = 0; br < rbl; ++br){
                psiPtr_offset[br] /= diagPtr_offset[br];
            }
        }
        
    }
}

#ifdef __ARM_FEATURE_SVE
void Foam::divMatrix::Jacobi_split_unroll32_sve
(
    scalarField& psi,
    const scalarField& bPrime
) const
{
    scalarField psiCopy(row_);

    const scalar* const __restrict__ bPrimePtr = bPrime.begin();

    scalar* __restrict__ psiCopyPtr = psiCopy.begin();
    scalar* __restrict__ psiPtr = psi.begin();

    const scalar* const __restrict__ diagPtr = diag_value_.begin();
    const scalar* const __restrict__ off_diag_value_Ptr = off_diag_value_.begin();
    svbool_t ptrue = svptrue_b64();
    svint64_t zero = svdup_s64(static_cast<label>(0));
    svint64_t max_row = svdup_s64(static_cast<label>(row_ - 1));

    #pragma omp parallel 
    {
        #pragma omp for
        for(label row = 0; row < row_; ++row){
            psiCopyPtr[row] = psiPtr[row];
        }
        #pragma omp for
        for(label bi = 0; bi < head_block_count_; ++bi){
            label rbs = DIV_BLOCK_START(bi);
            scalar* __restrict__ psiPtr_offset = psiPtr + rbs;
            const scalar* const __restrict__ bPrimePtr_offset = bPrimePtr + rbs;
            const scalar* const __restrict__ diagPtr_offset = diagPtr + rbs;
            svfloat64_t vpsi0, vpsi1, vpsi2, vpsi3;
            svfloat64_t tmp0, tmp1, tmp2, tmp3;
            svfloat64_t vdiag0, vdiag1, vdiag2, vdiag3;
            svfloat64_t vValue0, vValue1, vValue2, vValue3;
            svint64_t vIndex0, vIndex1, vIndex2, vIndex3;
            svfloat64_t vpsiCopy0, vpsiCopy1, vpsiCopy2, vpsiCopy3;
            vpsi0 = svld1_vnum(ptrue, bPrimePtr_offset, 0);
            vpsi1 = svld1_vnum(ptrue, bPrimePtr_offset, 1);
            vpsi2 = svld1_vnum(ptrue, bPrimePtr_offset, 2);
            vpsi3 = svld1_vnum(ptrue, bPrimePtr_offset, 3);

            label index_block_start = DIV_INDEX_BLOCK_START(rbs);
            for(label divcol = 0; divcol < distance_count_; ++divcol){
                label index_divcol_start = index_block_start + DIV_COL_OFFSET(divcol);
                label distance = distance_list_[divcol];
                if(distance + rbs >= 0){
                    vValue0 = svld1_vnum(ptrue, off_diag_value_Ptr + index_divcol_start, 0);
                    vValue1 = svld1_vnum(ptrue, off_diag_value_Ptr + index_divcol_start, 1);
                    vValue2 = svld1_vnum(ptrue, off_diag_value_Ptr + index_divcol_start, 2);
                    vValue3 = svld1_vnum(ptrue, off_diag_value_Ptr + index_divcol_start, 3);
                    vpsiCopy0 = svld1_vnum(ptrue, psiCopyPtr + distance + rbs, 0);
                    vpsiCopy1 = svld1_vnum(ptrue, psiCopyPtr + distance + rbs, 1);
                    vpsiCopy2 = svld1_vnum(ptrue, psiCopyPtr + distance + rbs, 2);
                    vpsiCopy3 = svld1_vnum(ptrue, psiCopyPtr + distance + rbs, 3);
                    tmp0 = svmul_z(ptrue, vValue0, vpsiCopy0);
                    tmp1 = svmul_z(ptrue, vValue1, vpsiCopy1);
                    tmp2 = svmul_z(ptrue, vValue2, vpsiCopy2);
                    tmp3 = svmul_z(ptrue, vValue3, vpsiCopy3);
                    vpsi0 = svsub_z(ptrue, vpsi0, tmp0);
                    vpsi1 = svsub_z(ptrue, vpsi1, tmp1);
                    vpsi2 = svsub_z(ptrue, vpsi2, tmp2);
                    vpsi3 = svsub_z(ptrue, vpsi3, tmp3);
                }else{
                    vValue0 = svld1_vnum(ptrue, off_diag_value_Ptr + index_divcol_start, 0);
                    vValue1 = svld1_vnum(ptrue, off_diag_value_Ptr + index_divcol_start, 1);
                    vValue2 = svld1_vnum(ptrue, off_diag_value_Ptr + index_divcol_start, 2);
                    vValue3 = svld1_vnum(ptrue, off_diag_value_Ptr + index_divcol_start, 3);
                    vIndex0 = svindex_s64(distance + rbs + 0, 1);
                    vIndex1 = svindex_s64(distance + rbs + 8, 1);
                    vIndex2 = svindex_s64(distance + rbs + 16, 1);
                    vIndex3 = svindex_s64(distance + rbs + 24, 1);
                    vIndex0 = svmax_z(ptrue, vIndex0, zero);
                    vIndex1 = svmax_z(ptrue, vIndex1, zero);
                    vIndex2 = svmax_z(ptrue, vIndex2, zero);
                    vIndex3 = svmax_z(ptrue, vIndex3, zero);
                    vpsiCopy0 = svld1_gather_index(ptrue, psiCopyPtr, vIndex0);
                    vpsiCopy1 = svld1_gather_index(ptrue, psiCopyPtr, vIndex1);
                    vpsiCopy2 = svld1_gather_index(ptrue, psiCopyPtr, vIndex2);
                    vpsiCopy3 = svld1_gather_index(ptrue, psiCopyPtr, vIndex3);
                    tmp0 = svmul_z(ptrue, vValue0, vpsiCopy0);
                    tmp1 = svmul_z(ptrue, vValue1, vpsiCopy1);
                    tmp2 = svmul_z(ptrue, vValue2, vpsiCopy2);
                    tmp3 = svmul_z(ptrue, vValue3, vpsiCopy3);
                    vpsi0 = svsub_z(ptrue, vpsi0, tmp0);
                    vpsi1 = svsub_z(ptrue, vpsi1, tmp1);
                    vpsi2 = svsub_z(ptrue, vpsi2, tmp2);
                    vpsi3 = svsub_z(ptrue, vpsi3, tmp3);
                }
            }
            vdiag0 = svld1_vnum(ptrue, diagPtr_offset, 0);
            vdiag1 = svld1_vnum(ptrue, diagPtr_offset, 1);
            vdiag2 = svld1_vnum(ptrue, diagPtr_offset, 2);
            vdiag3 = svld1_vnum(ptrue, diagPtr_offset, 3);
            vpsi0 = svdiv_z(ptrue, vpsi0, vdiag0);
            vpsi1 = svdiv_z(ptrue, vpsi1, vdiag1);
            vpsi2 = svdiv_z(ptrue, vpsi2, vdiag2);
            vpsi3 = svdiv_z(ptrue, vpsi3, vdiag3);
            svst1_vnum(ptrue, psiPtr_offset, 0, vpsi0);
            svst1_vnum(ptrue, psiPtr_offset, 1, vpsi1);
            svst1_vnum(ptrue, psiPtr_offset, 2, vpsi2);
            svst1_vnum(ptrue, psiPtr_offset, 3, vpsi3);
        }
        #pragma omp for
        for(label bi = head_block_count_; bi < block_count_ - tail_block_count_; ++bi){
            label rbs = DIV_BLOCK_START(bi);
            scalar* __restrict__ psiPtr_offset = psiPtr + rbs;
            const scalar* const __restrict__ bPrimePtr_offset = bPrimePtr + rbs;
            const scalar* const __restrict__ diagPtr_offset = diagPtr + rbs;
            svfloat64_t vpsi0, vpsi1, vpsi2, vpsi3;
            svfloat64_t tmp0, tmp1, tmp2, tmp3;
            svfloat64_t vdiag0, vdiag1, vdiag2, vdiag3;
            svfloat64_t vValue0, vValue1, vValue2, vValue3;
            svint64_t vIndex0, vIndex1, vIndex2, vIndex3;
            svfloat64_t vpsiCopy0, vpsiCopy1, vpsiCopy2, vpsiCopy3;
            vpsi0 = svld1_vnum(ptrue, bPrimePtr_offset, 0);
            vpsi1 = svld1_vnum(ptrue, bPrimePtr_offset, 1);
            vpsi2 = svld1_vnum(ptrue, bPrimePtr_offset, 2);
            vpsi3 = svld1_vnum(ptrue, bPrimePtr_offset, 3);
            
            label index_block_start = DIV_INDEX_BLOCK_START(rbs);
            for(label divcol = 0; divcol < distance_count_; ++divcol){
                label index_divcol_start = index_block_start + DIV_COL_OFFSET(divcol);
                label distance = distance_list_[divcol];
                vValue0 = svld1_vnum(ptrue, off_diag_value_Ptr + index_divcol_start, 0);
                vValue1 = svld1_vnum(ptrue, off_diag_value_Ptr + index_divcol_start, 1);
                vValue2 = svld1_vnum(ptrue, off_diag_value_Ptr + index_divcol_start, 2);
                vValue3 = svld1_vnum(ptrue, off_diag_value_Ptr + index_divcol_start, 3);
                vpsiCopy0 = svld1_vnum(ptrue, psiCopyPtr + distance + rbs, 0);
                vpsiCopy1 = svld1_vnum(ptrue, psiCopyPtr + distance + rbs, 1);
                vpsiCopy2 = svld1_vnum(ptrue, psiCopyPtr + distance + rbs, 2);
                vpsiCopy3 = svld1_vnum(ptrue, psiCopyPtr + distance + rbs, 3);
                tmp0 = svmul_z(ptrue, vValue0, vpsiCopy0);
                tmp1 = svmul_z(ptrue, vValue1, vpsiCopy1);
                tmp2 = svmul_z(ptrue, vValue2, vpsiCopy2);
                tmp3 = svmul_z(ptrue, vValue3, vpsiCopy3);
                vpsi0 = svsub_z(ptrue, vpsi0, tmp0);
                vpsi1 = svsub_z(ptrue, vpsi1, tmp1);
                vpsi2 = svsub_z(ptrue, vpsi2, tmp2);
                vpsi3 = svsub_z(ptrue, vpsi3, tmp3);
            }
            vdiag0 = svld1_vnum(ptrue, diagPtr_offset, 0);
            vdiag1 = svld1_vnum(ptrue, diagPtr_offset, 1);
            vdiag2 = svld1_vnum(ptrue, diagPtr_offset, 2);
            vdiag3 = svld1_vnum(ptrue, diagPtr_offset, 3);
            vpsi0 = svdiv_z(ptrue, vpsi0, vdiag0);
            vpsi1 = svdiv_z(ptrue, vpsi1, vdiag1);
            vpsi2 = svdiv_z(ptrue, vpsi2, vdiag2);
            vpsi3 = svdiv_z(ptrue, vpsi3, vdiag3);
            svst1_vnum(ptrue, psiPtr_offset, 0, vpsi0);
            svst1_vnum(ptrue, psiPtr_offset, 1, vpsi1);
            svst1_vnum(ptrue, psiPtr_offset, 2, vpsi2);
            svst1_vnum(ptrue, psiPtr_offset, 3, vpsi3);
        }
        #pragma omp for
        for(label bi = block_count_ - tail_block_count_; bi < block_count_; ++bi){
            label rbs = DIV_BLOCK_START(bi);
            label rbe = DIV_BLOCK_END(rbs);
            scalar* __restrict__ psiPtr_offset = psiPtr + rbs;
            const scalar* const __restrict__ bPrimePtr_offset = bPrimePtr + rbs;
            const scalar* const __restrict__ diagPtr_offset = diagPtr + rbs;
            svfloat64_t vpsi0, vpsi1, vpsi2, vpsi3;
            svfloat64_t tmp0, tmp1, tmp2, tmp3;
            svfloat64_t vdiag0, vdiag1, vdiag2, vdiag3;
            svfloat64_t vValue0, vValue1, vValue2, vValue3;
            svint64_t vIndex0, vIndex1, vIndex2, vIndex3;
            svfloat64_t vpsiCopy0, vpsiCopy1, vpsiCopy2, vpsiCopy3;
            vpsi0 = svld1_vnum(ptrue, bPrimePtr_offset, 0);
            vpsi1 = svld1_vnum(ptrue, bPrimePtr_offset, 1);
            vpsi2 = svld1_vnum(ptrue, bPrimePtr_offset, 2);
            vpsi3 = svld1_vnum(ptrue, bPrimePtr_offset, 3);

            label index_block_start = DIV_INDEX_BLOCK_START(rbs);
            for(label divcol = 0; divcol < distance_count_; ++divcol){
                label index_divcol_start = index_block_start + DIV_COL_OFFSET(divcol);
                label distance = distance_list_[divcol];
                if(distance + rbe <= row_){
                    vValue0 = svld1_vnum(ptrue, off_diag_value_Ptr + index_divcol_start, 0);
                    vValue1 = svld1_vnum(ptrue, off_diag_value_Ptr + index_divcol_start, 1);
                    vValue2 = svld1_vnum(ptrue, off_diag_value_Ptr + index_divcol_start, 2);
                    vValue3 = svld1_vnum(ptrue, off_diag_value_Ptr + index_divcol_start, 3);
                    vpsiCopy0 = svld1_vnum(ptrue, psiCopyPtr + distance + rbs, 0);
                    vpsiCopy1 = svld1_vnum(ptrue, psiCopyPtr + distance + rbs, 1);
                    vpsiCopy2 = svld1_vnum(ptrue, psiCopyPtr + distance + rbs, 2);
                    vpsiCopy3 = svld1_vnum(ptrue, psiCopyPtr + distance + rbs, 3);
                    tmp0 = svmul_z(ptrue, vValue0, vpsiCopy0);
                    tmp1 = svmul_z(ptrue, vValue1, vpsiCopy1);
                    tmp2 = svmul_z(ptrue, vValue2, vpsiCopy2);
                    tmp3 = svmul_z(ptrue, vValue3, vpsiCopy3);
                    vpsi0 = svsub_z(ptrue, vpsi0, tmp0);
                    vpsi1 = svsub_z(ptrue, vpsi1, tmp1);
                    vpsi2 = svsub_z(ptrue, vpsi2, tmp2);
                    vpsi3 = svsub_z(ptrue, vpsi3, tmp3);
                }else{
                    vValue0 = svld1_vnum(ptrue, off_diag_value_Ptr + index_divcol_start, 0);
                    vValue1 = svld1_vnum(ptrue, off_diag_value_Ptr + index_divcol_start, 1);
                    vValue2 = svld1_vnum(ptrue, off_diag_value_Ptr + index_divcol_start, 2);
                    vValue3 = svld1_vnum(ptrue, off_diag_value_Ptr + index_divcol_start, 3);
                    vIndex0 = svindex_s64(distance + rbs + 0, 1);
                    vIndex1 = svindex_s64(distance + rbs + 8, 1);
                    vIndex2 = svindex_s64(distance + rbs + 16, 1);
                    vIndex3 = svindex_s64(distance + rbs + 24, 1);
                    vIndex0 = svmin_z(ptrue, vIndex0, max_row);
                    vIndex1 = svmin_z(ptrue, vIndex1, max_row);
                    vIndex2 = svmin_z(ptrue, vIndex2, max_row);
                    vIndex3 = svmin_z(ptrue, vIndex3, max_row);
                    vpsiCopy0 = svld1_gather_index(ptrue, psiCopyPtr, vIndex0);
                    vpsiCopy1 = svld1_gather_index(ptrue, psiCopyPtr, vIndex1);
                    vpsiCopy2 = svld1_gather_index(ptrue, psiCopyPtr, vIndex2);
                    vpsiCopy3 = svld1_gather_index(ptrue, psiCopyPtr, vIndex3);
                    tmp0 = svmul_z(ptrue, vValue0, vpsiCopy0);
                    tmp1 = svmul_z(ptrue, vValue1, vpsiCopy1);
                    tmp2 = svmul_z(ptrue, vValue2, vpsiCopy2);
                    tmp3 = svmul_z(ptrue, vValue3, vpsiCopy3);
                    vpsi0 = svsub_z(ptrue, vpsi0, tmp0);
                    vpsi1 = svsub_z(ptrue, vpsi1, tmp1);
                    vpsi2 = svsub_z(ptrue, vpsi2, tmp2);
                    vpsi3 = svsub_z(ptrue, vpsi3, tmp3);
                }
            }
            vdiag0 = svld1_vnum(ptrue, diagPtr_offset, 0);
            vdiag1 = svld1_vnum(ptrue, diagPtr_offset, 1);
            vdiag2 = svld1_vnum(ptrue, diagPtr_offset, 2);
            vdiag3 = svld1_vnum(ptrue, diagPtr_offset, 3);
            vpsi0 = svdiv_z(ptrue, vpsi0, vdiag0);
            vpsi1 = svdiv_z(ptrue, vpsi1, vdiag1);
            vpsi2 = svdiv_z(ptrue, vpsi2, vdiag2);
            vpsi3 = svdiv_z(ptrue, vpsi3, vdiag3);
            svst1_vnum(ptrue, psiPtr_offset, 0, vpsi0);
            svst1_vnum(ptrue, psiPtr_offset, 1, vpsi1);
            svst1_vnum(ptrue, psiPtr_offset, 2, vpsi2);
            svst1_vnum(ptrue, psiPtr_offset, 3, vpsi3);
        }
    }
}
#endif

void Foam::divMatrix::Jacobi_split
(
    scalarField& psi,
    const scalarField& bPrime
) const
{
    scalarField psiCopy(row_);

    const scalar* const __restrict__ bPrimePtr = bPrime.begin();

    scalar* __restrict__ psiCopyPtr = psiCopy.begin();
    scalar* __restrict__ psiPtr = psi.begin();

    const scalar* const __restrict__ diagPtr = diag_value_.begin();
    const scalar* const __restrict__ off_diag_value_Ptr = off_diag_value_.begin();

    #pragma omp parallel 
    {
        #pragma omp for
        for(label row = 0; row < row_; ++row){
            psiCopyPtr[row] = psiPtr[row];
        }
        #pragma omp for
        for(label bi = 0; bi < head_block_count_; ++bi){
            label rbs = DIV_BLOCK_START(bi);
            label rbe = DIV_BLOCK_END(rbs);
            label rbl = DIV_BLOCK_LEN(rbs,rbe);
            scalar* __restrict__ psiPtr_offset = psiPtr + rbs;
            const scalar* const __restrict__ bPrimePtr_offset = bPrimePtr + rbs;
            const scalar* const __restrict__ diagPtr_offset = diagPtr + rbs;
            for(label br = 0; br < rbl; ++br){
                psiPtr_offset[br] = bPrimePtr_offset[br];
            }
            label index_block_start = DIV_INDEX_BLOCK_START(rbs);
            for(label divcol = 0; divcol < distance_count_; ++divcol){
                label index_divcol_start = index_block_start + DIV_COL_OFFSET(divcol);
                label distance = distance_list_[divcol];
                if(distance + rbs >= 0){
                    for(label br = 0; br < rbl; ++br){
                        psiPtr_offset[br] -= off_diag_value_Ptr[index_divcol_start + br] * psiCopyPtr[distance + rbs + br];
                    }
                }else{
                    for(label br = 0; br < rbl; ++br){
                        label col = distance + rbs + br;
                        col = std::max(col, static_cast<label>(0));
                        psiPtr_offset[br] -= off_diag_value_Ptr[index_divcol_start + br] * psiCopyPtr[col];
                    }
                }
            }
            for(label br = 0; br < rbl; ++br){
                psiPtr_offset[br] /= diagPtr_offset[br];
            }
        }
        #pragma omp for
        for(label bi = head_block_count_; bi < block_count_ - tail_block_count_; ++bi){
            label rbs = DIV_BLOCK_START(bi);
            label rbe = DIV_BLOCK_END(rbs);
            label rbl = DIV_BLOCK_LEN(rbs,rbe);
            scalar* __restrict__ psiPtr_offset = psiPtr + rbs;
            const scalar* const __restrict__ bPrimePtr_offset = bPrimePtr + rbs;
            const scalar* const __restrict__ diagPtr_offset = diagPtr + rbs;
            for(label br = 0; br < rbl; ++br){
                psiPtr_offset[br] = bPrimePtr_offset[br];
            }
            label index_block_start = DIV_INDEX_BLOCK_START(rbs);
            for(label divcol = 0; divcol < distance_count_; ++divcol){
                label index_divcol_start = index_block_start + DIV_COL_OFFSET(divcol);
                label distance = distance_list_[divcol];
                for(label br = 0; br < rbl; ++br){
                    psiPtr_offset[br] -= off_diag_value_Ptr[index_divcol_start + br] * psiCopyPtr[distance + rbs + br];
                }
            }
            for(label br = 0; br < rbl; ++br){
                psiPtr_offset[br] /= diagPtr_offset[br];
            }
        }
        #pragma omp for
        for(label bi = block_count_ - tail_block_count_; bi < block_count_; ++bi){
            label rbs = DIV_BLOCK_START(bi);
            label rbe = DIV_BLOCK_END(rbs);
            label rbl = DIV_BLOCK_LEN(rbs,rbe);
            scalar* __restrict__ psiPtr_offset = psiPtr + rbs;
            const scalar* const __restrict__ bPrimePtr_offset = bPrimePtr + rbs;
            const scalar* const __restrict__ diagPtr_offset = diagPtr + rbs;
            for(label br = 0; br < rbl; ++br){
                psiPtr_offset[br] = bPrimePtr_offset[br];
            }
            label index_block_start = DIV_INDEX_BLOCK_START(rbs);
            for(label divcol = 0; divcol < distance_count_; ++divcol){
                label index_divcol_start = index_block_start + DIV_COL_OFFSET(divcol);
                label distance = distance_list_[divcol];
                if(distance + rbe <= row_){
                    for(label br = 0; br < rbl; ++br){
                        psiPtr_offset[br] -= off_diag_value_Ptr[index_divcol_start + br] * psiCopyPtr[distance + rbs + br];
                    }
                }else{
                    for(label br = 0; br < rbl; ++br){
                        label col = distance + rbs + br;
                        col = std::min(col, row_ - 1);
                        psiPtr_offset[br] -= off_diag_value_Ptr[index_divcol_start + br] * psiCopyPtr[col];
                    }
                }
            }
            for(label br = 0; br < rbl; ++br){
                psiPtr_offset[br] /= diagPtr_offset[br];
            }
        }
    }
}

// ************************************************************************* //
