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
#include <typeinfo>
#include <omp.h>
#include <algorithm>
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h> 
#endif
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void Foam::divMatrix::SpMV
(
    scalarField& Apsi,
    const scalarField& psi
) const
{
#ifdef __ARM_FEATURE_SVE
    assert(svcntd() == 8);
#endif
    if(block_count_ > 1){
        if(row_block_size_ == 32){
#ifdef __ARM_FEATURE_SVE
            SpMV_split_unroll32_sve(Apsi, psi);
#else
            SpMV_split(Apsi, psi);
#endif
        }else{
            SpMV_split(Apsi, psi);
        }
    }else{
        SpMV_naive(Apsi, psi);
    }

    flops_ += row_ * (distance_count_ + 1) * 2.;
}

void Foam::divMatrix::SpMV_naive
(
    scalarField& Apsi,
    const scalarField& psi
) const
{
    scalar* ApsiPtr = Apsi.begin();
    const scalar* const __restrict__ psiPtr = psi.begin();
    const scalar* const __restrict__ diagPtr = diag_value_.begin();
    const scalar* const __restrict__ off_diag_value_Ptr = off_diag_value_.begin();

    #pragma omp parallel for
    for(label bi = 0; bi < block_count_; ++bi){
        label rbs = DIV_BLOCK_START(bi);
        label rbe = DIV_BLOCK_END(rbs);
        label rbl = DIV_BLOCK_LEN(rbs,rbe);
        scalar* ApsiPtr_offset = ApsiPtr + rbs;
        const scalar* const __restrict__ diagPtr_offset = diagPtr + rbs;
        for(label br = 0; br < rbl; ++br){
            ApsiPtr_offset[br] = diagPtr_offset[br] * psiPtr[rbs + br];
        }
        label index_block_start = DIV_INDEX_BLOCK_START(rbs);
        for(label divcol = 0; divcol < distance_count_; ++divcol){
            label index_divcol_start = index_block_start + DIV_COL_OFFSET(divcol);
            label distance = distance_list_[divcol];
            if(distance + rbs >= 0 && distance + rbe <= row_){
                for(label br = 0; br < rbl; ++br){
                    ApsiPtr_offset[br] += off_diag_value_Ptr[index_divcol_start + br] * psiPtr[distance + rbs + br];
                }
            }else{
                for(label br = 0; br < rbl; ++br){
                    label col = distance + rbs + br;
                    col = std::max(col, static_cast<label>(0));
                    col = std::min(col, row_ - 1);
                    ApsiPtr_offset[br] += off_diag_value_Ptr[index_divcol_start + br] * psiPtr[col];
                }       
            }
        }
    }
}

#ifdef __ARM_FEATURE_SVE
void Foam::divMatrix::SpMV_split_unroll32_sve
(
    scalarField& Apsi,
    const scalarField& psi
) const
{
    scalar* ApsiPtr = Apsi.begin();
    const scalar* const __restrict__ psiPtr = psi.begin();
    const scalar* const __restrict__ diagPtr = diag_value_.begin();
    const scalar* const __restrict__ off_diag_value_Ptr = off_diag_value_.begin();
    svbool_t ptrue = svptrue_b64();
    svint64_t zero = svdup_s64(static_cast<label>(0));
    svint64_t max_row = svdup_s64(static_cast<label>(row_ - 1));

    #pragma omp parallel
    {
    #pragma omp for
    for(label bi = 0; bi < head_block_count_; ++bi){
        label rbs = DIV_BLOCK_START(bi);
        scalar* ApsiPtr_offset = ApsiPtr + rbs;
        const scalar* const __restrict__ diagPtr_offset = diagPtr + rbs;
        svfloat64_t vApsi0, vApsi1, vApsi2, vApsi3;
        svfloat64_t vpsi0, vpsi1, vpsi2, vpsi3;
        svfloat64_t vdiag0, vdiag1, vdiag2, vdiag3;
        svfloat64_t tmp0, tmp1, tmp2, tmp3;
        svint64_t vIndex0, vIndex1, vIndex2, vIndex3;
        svfloat64_t vValue0, vValue1, vValue2, vValue3;
        vdiag0 = svld1_vnum(ptrue, diagPtr_offset, 0);
        vdiag1 = svld1_vnum(ptrue, diagPtr_offset, 1);
        vdiag2 = svld1_vnum(ptrue, diagPtr_offset, 2);
        vdiag3 = svld1_vnum(ptrue, diagPtr_offset, 3);
        vpsi0 = svld1_vnum(ptrue, psiPtr + rbs, 0);
        vpsi1 = svld1_vnum(ptrue, psiPtr + rbs, 1);
        vpsi2 = svld1_vnum(ptrue, psiPtr + rbs, 2);
        vpsi3 = svld1_vnum(ptrue, psiPtr + rbs, 3);
        vApsi0 = svmul_z(ptrue, vdiag0, vpsi0);
        vApsi1 = svmul_z(ptrue, vdiag1, vpsi1);
        vApsi2 = svmul_z(ptrue, vdiag2, vpsi2);
        vApsi3 = svmul_z(ptrue, vdiag3, vpsi3);
        label index_block_start = DIV_INDEX_BLOCK_START(rbs);
        for(label divcol = 0; divcol < distance_count_; ++divcol){
            label index_divcol_start = index_block_start + DIV_COL_OFFSET(divcol);
            label distance = distance_list_[divcol];
            if(distance + rbs >= 0){
                vpsi0 = svld1_vnum(ptrue, psiPtr + distance + rbs, 0);
                vpsi1 = svld1_vnum(ptrue, psiPtr + distance + rbs, 1);
                vpsi2 = svld1_vnum(ptrue, psiPtr + distance + rbs, 2);
                vpsi3 = svld1_vnum(ptrue, psiPtr + distance + rbs, 3);
                vValue0 = svld1_vnum(ptrue, off_diag_value_Ptr + index_divcol_start, 0);
                vValue1 = svld1_vnum(ptrue, off_diag_value_Ptr + index_divcol_start, 1);
                vValue2 = svld1_vnum(ptrue, off_diag_value_Ptr + index_divcol_start, 2);
                vValue3 = svld1_vnum(ptrue, off_diag_value_Ptr + index_divcol_start, 3);
                tmp0 = svmul_z(ptrue, vValue0, vpsi0);
                tmp1 = svmul_z(ptrue, vValue1, vpsi1);
                tmp2 = svmul_z(ptrue, vValue2, vpsi2);
                tmp3 = svmul_z(ptrue, vValue3, vpsi3);
                vApsi0 = svadd_z(ptrue, tmp0, vApsi0);
                vApsi1 = svadd_z(ptrue, tmp1, vApsi1);
                vApsi2 = svadd_z(ptrue, tmp2, vApsi2);
                vApsi3 = svadd_z(ptrue, tmp3, vApsi3);
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
                vpsi0 = svld1_gather_index(ptrue, psiPtr, vIndex0);
                vpsi1 = svld1_gather_index(ptrue, psiPtr, vIndex1);
                vpsi2 = svld1_gather_index(ptrue, psiPtr, vIndex2);
                vpsi3 = svld1_gather_index(ptrue, psiPtr, vIndex3);
                tmp0 = svmul_z(ptrue, vValue0, vpsi0);
                tmp1 = svmul_z(ptrue, vValue1, vpsi1);
                tmp2 = svmul_z(ptrue, vValue2, vpsi2);
                tmp3 = svmul_z(ptrue, vValue3, vpsi3);
                vApsi0 = svadd_z(ptrue, tmp0, vApsi0);
                vApsi1 = svadd_z(ptrue, tmp1, vApsi1);
                vApsi2 = svadd_z(ptrue, tmp2, vApsi2);
                vApsi3 = svadd_z(ptrue, tmp3, vApsi3);
            }
        }
        svst1_vnum(ptrue, ApsiPtr_offset, 0, vApsi0);
        svst1_vnum(ptrue, ApsiPtr_offset, 1, vApsi1);
        svst1_vnum(ptrue, ApsiPtr_offset, 2, vApsi2);
        svst1_vnum(ptrue, ApsiPtr_offset, 3, vApsi3);
    }
    
    #pragma omp for
    for(label bi = head_block_count_; bi < block_count_ - tail_block_count_; ++bi){
        label rbs = DIV_BLOCK_START(bi);
        scalar* ApsiPtr_offset = ApsiPtr + rbs;
        const scalar* const __restrict__ diagPtr_offset = diagPtr + rbs;
        svfloat64_t vApsi0, vApsi1, vApsi2, vApsi3;
        svfloat64_t vpsi0, vpsi1, vpsi2, vpsi3;
        svfloat64_t vdiag0, vdiag1, vdiag2, vdiag3;
        svfloat64_t tmp0, tmp1, tmp2, tmp3;
        svint64_t vIndex0, vIndex1, vIndex2, vIndex3;
        svfloat64_t vValue0, vValue1, vValue2, vValue3;
        vdiag0 = svld1_vnum(ptrue, diagPtr_offset, 0);
        vdiag1 = svld1_vnum(ptrue, diagPtr_offset, 1);
        vdiag2 = svld1_vnum(ptrue, diagPtr_offset, 2);
        vdiag3 = svld1_vnum(ptrue, diagPtr_offset, 3);
        vpsi0 = svld1_vnum(ptrue, psiPtr + rbs, 0);
        vpsi1 = svld1_vnum(ptrue, psiPtr + rbs, 1);
        vpsi2 = svld1_vnum(ptrue, psiPtr + rbs, 2);
        vpsi3 = svld1_vnum(ptrue, psiPtr + rbs, 3);
        vApsi0 = svmul_z(ptrue, vdiag0, vpsi0);
        vApsi1 = svmul_z(ptrue, vdiag1, vpsi1);
        vApsi2 = svmul_z(ptrue, vdiag2, vpsi2);
        vApsi3 = svmul_z(ptrue, vdiag3, vpsi3);
        label index_block_start = DIV_INDEX_BLOCK_START(rbs);
        for(label divcol = 0; divcol < distance_count_; ++divcol){
            label index_divcol_start = index_block_start + DIV_COL_OFFSET(divcol);
            label distance = distance_list_[divcol];
            vpsi0 = svld1_vnum(ptrue, psiPtr + distance + rbs, 0);
            vpsi1 = svld1_vnum(ptrue, psiPtr + distance + rbs, 1);
            vpsi2 = svld1_vnum(ptrue, psiPtr + distance + rbs, 2);
            vpsi3 = svld1_vnum(ptrue, psiPtr + distance + rbs, 3);
            vValue0 = svld1_vnum(ptrue, off_diag_value_Ptr + index_divcol_start, 0);
            vValue1 = svld1_vnum(ptrue, off_diag_value_Ptr + index_divcol_start, 1);
            vValue2 = svld1_vnum(ptrue, off_diag_value_Ptr + index_divcol_start, 2);
            vValue3 = svld1_vnum(ptrue, off_diag_value_Ptr + index_divcol_start, 3);
            tmp0 = svmul_z(ptrue, vValue0, vpsi0);
            tmp1 = svmul_z(ptrue, vValue1, vpsi1);
            tmp2 = svmul_z(ptrue, vValue2, vpsi2);
            tmp3 = svmul_z(ptrue, vValue3, vpsi3);
            vApsi0 = svadd_z(ptrue, tmp0, vApsi0);
            vApsi1 = svadd_z(ptrue, tmp1, vApsi1);
            vApsi2 = svadd_z(ptrue, tmp2, vApsi2);
            vApsi3 = svadd_z(ptrue, tmp3, vApsi3);
        }
        svst1_vnum(ptrue, ApsiPtr_offset, 0, vApsi0);
        svst1_vnum(ptrue, ApsiPtr_offset, 1, vApsi1);
        svst1_vnum(ptrue, ApsiPtr_offset, 2, vApsi2);
        svst1_vnum(ptrue, ApsiPtr_offset, 3, vApsi3);
    }
    
    #pragma omp for
    for(label bi = block_count_ - tail_block_count_; bi < block_count_; ++bi){
        label rbs = DIV_BLOCK_START(bi);
        label rbe = DIV_BLOCK_END(rbs);
        scalar* ApsiPtr_offset = ApsiPtr + rbs;
        const scalar* const __restrict__ diagPtr_offset = diagPtr + rbs;
        svfloat64_t vApsi0, vApsi1, vApsi2, vApsi3;
        svfloat64_t vpsi0, vpsi1, vpsi2, vpsi3;
        svfloat64_t vdiag0, vdiag1, vdiag2, vdiag3;
        svfloat64_t tmp0, tmp1, tmp2, tmp3;
        svint64_t vIndex0, vIndex1, vIndex2, vIndex3;
        svfloat64_t vValue0, vValue1, vValue2, vValue3;
        vdiag0 = svld1_vnum(ptrue, diagPtr_offset, 0);
        vdiag1 = svld1_vnum(ptrue, diagPtr_offset, 1);
        vdiag2 = svld1_vnum(ptrue, diagPtr_offset, 2);
        vdiag3 = svld1_vnum(ptrue, diagPtr_offset, 3);
        vpsi0 = svld1_vnum(ptrue, psiPtr + rbs, 0);
        vpsi1 = svld1_vnum(ptrue, psiPtr + rbs, 1);
        vpsi2 = svld1_vnum(ptrue, psiPtr + rbs, 2);
        vpsi3 = svld1_vnum(ptrue, psiPtr + rbs, 3);
        vApsi0 = svmul_z(ptrue, vdiag0, vpsi0);
        vApsi1 = svmul_z(ptrue, vdiag1, vpsi1);
        vApsi2 = svmul_z(ptrue, vdiag2, vpsi2);
        vApsi3 = svmul_z(ptrue, vdiag3, vpsi3);
        label index_block_start = DIV_INDEX_BLOCK_START(rbs);
        for(label divcol = 0; divcol < distance_count_; ++divcol){
            label index_divcol_start = index_block_start + DIV_COL_OFFSET(divcol);
            label distance = distance_list_[divcol];
            if(distance + rbe <= row_){
                vpsi0 = svld1_vnum(ptrue, psiPtr + distance + rbs, 0);
                vpsi1 = svld1_vnum(ptrue, psiPtr + distance + rbs, 1);
                vpsi2 = svld1_vnum(ptrue, psiPtr + distance + rbs, 2);
                vpsi3 = svld1_vnum(ptrue, psiPtr + distance + rbs, 3);
                vValue0 = svld1_vnum(ptrue, off_diag_value_Ptr + index_divcol_start, 0);
                vValue1 = svld1_vnum(ptrue, off_diag_value_Ptr + index_divcol_start, 1);
                vValue2 = svld1_vnum(ptrue, off_diag_value_Ptr + index_divcol_start, 2);
                vValue3 = svld1_vnum(ptrue, off_diag_value_Ptr + index_divcol_start, 3);
                tmp0 = svmul_z(ptrue, vValue0, vpsi0);
                tmp1 = svmul_z(ptrue, vValue1, vpsi1);
                tmp2 = svmul_z(ptrue, vValue2, vpsi2);
                tmp3 = svmul_z(ptrue, vValue3, vpsi3);
                vApsi0 = svadd_z(ptrue, tmp0, vApsi0);
                vApsi1 = svadd_z(ptrue, tmp1, vApsi1);
                vApsi2 = svadd_z(ptrue, tmp2, vApsi2);
                vApsi3 = svadd_z(ptrue, tmp3, vApsi3);
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
                vpsi0 = svld1_gather_index(ptrue, psiPtr, vIndex0);
                vpsi1 = svld1_gather_index(ptrue, psiPtr, vIndex1);
                vpsi2 = svld1_gather_index(ptrue, psiPtr, vIndex2);
                vpsi3 = svld1_gather_index(ptrue, psiPtr, vIndex3);
                tmp0 = svmul_z(ptrue, vValue0, vpsi0);
                tmp1 = svmul_z(ptrue, vValue1, vpsi1);
                tmp2 = svmul_z(ptrue, vValue2, vpsi2);
                tmp3 = svmul_z(ptrue, vValue3, vpsi3);
                vApsi0 = svadd_z(ptrue, tmp0, vApsi0);
                vApsi1 = svadd_z(ptrue, tmp1, vApsi1);
                vApsi2 = svadd_z(ptrue, tmp2, vApsi2);
                vApsi3 = svadd_z(ptrue, tmp3, vApsi3);            
            }
        }
        svst1_vnum(ptrue, ApsiPtr_offset, 0, vApsi0);
        svst1_vnum(ptrue, ApsiPtr_offset, 1, vApsi1);
        svst1_vnum(ptrue, ApsiPtr_offset, 2, vApsi2);
        svst1_vnum(ptrue, ApsiPtr_offset, 3, vApsi3);
    }
    }
}
#endif

void Foam::divMatrix::SpMV_split
(
    scalarField& Apsi,
    const scalarField& psi
) const
{
    scalar* ApsiPtr = Apsi.begin();
    const scalar* const __restrict__ psiPtr = psi.begin();
    const scalar* const __restrict__ diagPtr = diag_value_.begin();
    const scalar* const __restrict__ off_diag_value_Ptr = off_diag_value_.begin();

    #pragma omp parallel for
    for(label bi = 0; bi < head_block_count_; ++bi){
        label rbs = DIV_BLOCK_START(bi);
        label rbe = DIV_BLOCK_END(rbs);
        label rbl = DIV_BLOCK_LEN(rbs,rbe);
        scalar* ApsiPtr_offset = ApsiPtr + rbs;
        const scalar* const __restrict__ diagPtr_offset = diagPtr + rbs;
        for(label br = 0; br < rbl; ++br){
            ApsiPtr_offset[br] = diagPtr_offset[br] * psiPtr[rbs + br];
        }
        label index_block_start = DIV_INDEX_BLOCK_START(rbs);
        for(label divcol = 0; divcol < distance_count_; ++divcol){
            label index_divcol_start = index_block_start + DIV_COL_OFFSET(divcol);
            label distance = distance_list_[divcol];
            if(distance + rbs >= 0){
                for(label br = 0; br < rbl; ++br){
                    ApsiPtr_offset[br] += off_diag_value_Ptr[index_divcol_start + br] * psiPtr[distance + rbs + br];
                }
            }else{
                for(label br = 0; br < rbl; ++br){
                    label col = distance + rbs + br;
                    col = std::max(col, static_cast<label>(0));
                    ApsiPtr_offset[br] += off_diag_value_Ptr[index_divcol_start + br] * psiPtr[col];
                }               
            }
        }
    }

    #pragma omp parallel for
    for(label bi = head_block_count_; bi < block_count_ - tail_block_count_; ++bi){
        label rbs = DIV_BLOCK_START(bi);
        label rbe = DIV_BLOCK_END(rbs);
        label rbl = DIV_BLOCK_LEN(rbs,rbe);
        scalar* ApsiPtr_offset = ApsiPtr + rbs;
        const scalar* const __restrict__ diagPtr_offset = diagPtr + rbs;
        for(label br = 0; br < rbl; ++br){
            ApsiPtr_offset[br] = diagPtr_offset[br] * psiPtr[rbs + br];
        }
        label index_block_start = DIV_INDEX_BLOCK_START(rbs);
        for(label divcol = 0; divcol < distance_count_; ++divcol){
            label index_divcol_start = index_block_start + DIV_COL_OFFSET(divcol);
            label distance = distance_list_[divcol];
            for(label br = 0; br < rbl; ++br){
                ApsiPtr_offset[br] += off_diag_value_Ptr[index_divcol_start + br] * psiPtr[distance + rbs + br];
            }
        }
    }

    #pragma omp parallel for
    for(label bi = block_count_ - tail_block_count_; bi < block_count_; ++bi){
        label rbs = DIV_BLOCK_START(bi);
        label rbe = DIV_BLOCK_END(rbs);
        label rbl = DIV_BLOCK_LEN(rbs,rbe);
        scalar* ApsiPtr_offset = ApsiPtr + rbs;
        const scalar* const __restrict__ diagPtr_offset = diagPtr + rbs;
        for(label br = 0; br < rbl; ++br){
            ApsiPtr_offset[br] = diagPtr_offset[br] * psiPtr[rbs + br];
        }
        label index_block_start = DIV_INDEX_BLOCK_START(rbs);
        for(label divcol = 0; divcol < distance_count_; ++divcol){
            label index_divcol_start = index_block_start + DIV_COL_OFFSET(divcol);
            label distance = distance_list_[divcol];
            if(distance + rbe <= row_){
                for(label br = 0; br < rbl; ++br){
                    ApsiPtr_offset[br] += off_diag_value_Ptr[index_divcol_start + br] * psiPtr[distance + rbs + br];
                }
            }else{
                for(label br = 0; br < rbl; ++br){
                    label col = distance + rbs + br;
                    col = std::min(col, row_ -1);
                    ApsiPtr_offset[br] += off_diag_value_Ptr[index_divcol_start + br] * psiPtr[col];
                }               
            }
        }
    }
}

    

// ************************************************************************* //
