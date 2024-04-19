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

#include "ellMatrix.H"
#include <cassert>
#include <typeinfo>
#include <omp.h>
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h> 
#endif
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void Foam::ellMatrix::SpMV
(
    scalarField& Apsi,
    const scalarField& psi
) const
{
    if(row_block_size_ % 8 == 0 && ELL_BLOCK_TAIL == 0){
        if(row_block_size_ == 32){
#ifdef __ARM_FEATURE_SVE
            SpMV_UNROOL32_SVE(Apsi, psi);
#else
            SpMV_UNROOL8(Apsi, psi);
#endif
        }else{
#ifdef __ARM_FEATURE_SVE
        SpMV_UNROOL8_SVE(Apsi, psi);
#else
        SpMV_UNROOL8(Apsi, psi);
#endif
        }

    }else{
        SpMV_naive(Apsi, psi);
    }
}

void Foam::ellMatrix::SpMV_naive
(
    scalarField& Apsi,
    const scalarField& psi
) const
{
    scalar* ApsiPtr = Apsi.begin();
    const scalar* const __restrict__ psiPtr = psi.begin();
    const scalar* const __restrict__ diagPtr = diag_value_.begin();

    const label* const __restrict__ off_diag_colidx_Ptr = off_diag_colidx_.begin();
    const scalar* const __restrict__ off_diag_value_Ptr = off_diag_value_.begin();

    #pragma omp parallel for
    for(label rbs = 0; rbs < row_; rbs += row_block_size_){
        label rbe = ELL_BLOCK_END(rbs);
        label rbl = ELL_BLOCK_LEN(rbs,rbe);
        scalar* ApsiPtr_offset = ApsiPtr + rbs;
        const scalar* const __restrict__ diagPtr_offset = diagPtr + rbs;
        for(label br = 0; br < rbl; ++br){
            ApsiPtr_offset[br] = diagPtr_offset[br] * psiPtr[rbs + br];
        }
        label index_block_start = ELL_INDEX_BLOCK_START(rbs);
        for(label ellcol = 0; ellcol < max_count_; ++ellcol){
            label index_ellcol_start = index_block_start + ELL_COL_OFFSET(ellcol);
            for(label br = 0; br < rbl; ++br){
                scalar value = off_diag_value_Ptr[index_ellcol_start + br];
                label col = off_diag_colidx_Ptr[index_ellcol_start + br];
                ApsiPtr_offset[br] += value * psiPtr[col];
            }
        }
    }
}

void Foam::ellMatrix::SpMV_UNROOL8
(
    scalarField& Apsi,
    const scalarField& psi
) const
{
    scalar* ApsiPtr = Apsi.begin();
    const scalar* const __restrict__ psiPtr = psi.begin();
    const scalar* const __restrict__ diagPtr = diag_value_.begin();

    const label* const __restrict__ off_diag_colidx_Ptr = off_diag_colidx_.begin();
    const scalar* const __restrict__ off_diag_value_Ptr = off_diag_value_.begin();

    #pragma omp parallel for
    for(label rbs = 0; rbs < row_; rbs += row_block_size_){
        label rbe = ELL_BLOCK_END(rbs);
        label rbl = ELL_BLOCK_LEN(rbs,rbe);
        scalar* ApsiPtr_offset = ApsiPtr + rbs;
        const scalar* const __restrict__ diagPtr_offset = diagPtr + rbs;
        for(label br = 0; br < rbl; br += 8){
            ApsiPtr_offset[br + 0] = diagPtr_offset[br + 0] * psiPtr[rbs + br + 0];
            ApsiPtr_offset[br + 1] = diagPtr_offset[br + 1] * psiPtr[rbs + br + 1];
            ApsiPtr_offset[br + 2] = diagPtr_offset[br + 2] * psiPtr[rbs + br + 2];
            ApsiPtr_offset[br + 3] = diagPtr_offset[br + 3] * psiPtr[rbs + br + 3];
            ApsiPtr_offset[br + 4] = diagPtr_offset[br + 4] * psiPtr[rbs + br + 4];
            ApsiPtr_offset[br + 5] = diagPtr_offset[br + 5] * psiPtr[rbs + br + 5];
            ApsiPtr_offset[br + 6] = diagPtr_offset[br + 6] * psiPtr[rbs + br + 6];
            ApsiPtr_offset[br + 7] = diagPtr_offset[br + 7] * psiPtr[rbs + br + 7];
        }
        label index_block_start = ELL_INDEX_BLOCK_START(rbs);
        for(label ellcol = 0; ellcol < max_count_; ++ellcol){
            label index_ellcol_start = index_block_start + ELL_COL_OFFSET(ellcol);
            for(label br = 0; br < rbl; br += 8){
                ApsiPtr_offset[br + 0] += off_diag_value_Ptr[index_ellcol_start + br + 0] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + br + 0]];
                ApsiPtr_offset[br + 1] += off_diag_value_Ptr[index_ellcol_start + br + 1] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + br + 1]];
                ApsiPtr_offset[br + 2] += off_diag_value_Ptr[index_ellcol_start + br + 2] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + br + 2]];
                ApsiPtr_offset[br + 3] += off_diag_value_Ptr[index_ellcol_start + br + 3] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + br + 3]];
                ApsiPtr_offset[br + 4] += off_diag_value_Ptr[index_ellcol_start + br + 4] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + br + 4]];
                ApsiPtr_offset[br + 5] += off_diag_value_Ptr[index_ellcol_start + br + 5] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + br + 5]];
                ApsiPtr_offset[br + 6] += off_diag_value_Ptr[index_ellcol_start + br + 6] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + br + 6]];
                ApsiPtr_offset[br + 7] += off_diag_value_Ptr[index_ellcol_start + br + 7] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + br + 7]];
            }
        }
    }
}

#ifdef __ARM_FEATURE_SVE
void Foam::ellMatrix::SpMV_UNROOL8_SVE
(
    scalarField& Apsi,
    const scalarField& psi
) const
{
    assert(typeid(label) == typeid(int64_t));
    assert(svcntd() == 8);

    scalar* ApsiPtr = Apsi.begin();
    const scalar* const __restrict__ psiPtr = psi.begin();
    const scalar* const __restrict__ diagPtr = diag_value_.begin();

    const label* const __restrict__ off_diag_colidx_Ptr = off_diag_colidx_.begin();
    const scalar* const __restrict__ off_diag_value_Ptr = off_diag_value_.begin();
    svbool_t ptrue = svptrue_b64();

    if(block_count_ > 12){
        #pragma omp parallel 
        {
            #pragma omp for
            for(label bi = 0; bi < block_count_; ++bi){
                label rbs = ELL_BLOCK_START(bi);
                label rbe = ELL_BLOCK_END(rbs);
                label rbl = ELL_BLOCK_LEN(rbs,rbe);
                scalar* ApsiPtr_offset = ApsiPtr + rbs;
                const scalar* const __restrict__ diagPtr_offset = diagPtr + rbs;
                for(label br = 0; br < rbl; br += svcntd()){
                    svfloat64_t vdiag = svld1(ptrue, diagPtr_offset + br);
                    svfloat64_t vpsi = svld1(ptrue, psiPtr + rbs + br);
                    svfloat64_t vApsi = svmul_z(ptrue, vdiag, vpsi);
                    svst1(ptrue, ApsiPtr_offset + br, vApsi);
                }
                label index_block_start = ELL_INDEX_BLOCK_START(rbs);
                for(label ellcol = 0; ellcol < max_count_; ++ellcol){
                    label index_ellcol_start = index_block_start + ELL_COL_OFFSET(ellcol);
                    for(label br = 0; br < rbl; br += svcntd()){
                        svint64_t vIndex = svld1(ptrue, off_diag_colidx_Ptr + index_ellcol_start + br);
                        svfloat64_t vpsi = svld1_gather_index(ptrue, psiPtr, vIndex);
                        svfloat64_t vApsi = svld1(ptrue, ApsiPtr_offset + br);
                        svfloat64_t vValue = svld1(ptrue, off_diag_value_Ptr + index_ellcol_start + br);
                        // svfloat64_t vApsi_o = svmla_z(ptrue, vApsi, vValue, vpsi);
                        // svfloat64_t vApsi_o = svmad_z(ptrue, vValue, vpsi, vApsi);
                        svfloat64_t tmp = svmul_z(ptrue, vValue, vpsi);
                        vApsi = svadd_z(ptrue, tmp, vApsi);
                        svst1(ptrue, ApsiPtr_offset + br, vApsi);
                    }
                }
            }
        }
    }else{
        for(label bi = 0; bi < block_count_; ++bi){
            label rbs = ELL_BLOCK_START(bi);
            label rbe = ELL_BLOCK_END(rbs);
            label rbl = ELL_BLOCK_LEN(rbs,rbe);
            scalar* ApsiPtr_offset = ApsiPtr + rbs;
            const scalar* const __restrict__ diagPtr_offset = diagPtr + rbs;
            for(label br = 0; br < rbl; br += svcntd()){
                svfloat64_t vdiag = svld1(ptrue, diagPtr_offset + br);
                svfloat64_t vpsi = svld1(ptrue, psiPtr + rbs + br);
                svfloat64_t vApsi = svmul_z(ptrue, vdiag, vpsi);
                svst1(ptrue, ApsiPtr_offset + br, vApsi);
            }
            label index_block_start = ELL_INDEX_BLOCK_START(rbs);
            for(label ellcol = 0; ellcol < max_count_; ++ellcol){
                label index_ellcol_start = index_block_start + ELL_COL_OFFSET(ellcol);
                for(label br = 0; br < rbl; br += svcntd()){
                    svint64_t vIndex = svld1(ptrue, off_diag_colidx_Ptr + index_ellcol_start + br);
                    svfloat64_t vpsi = svld1_gather_index(ptrue, psiPtr, vIndex);
                    svfloat64_t vApsi = svld1(ptrue, ApsiPtr_offset + br);
                    svfloat64_t vValue = svld1(ptrue, off_diag_value_Ptr + index_ellcol_start + br);
                    // svfloat64_t vApsi_o = svmla_z(ptrue, vApsi, vValue, vpsi);
                    // svfloat64_t vApsi_o = svmad_z(ptrue, vValue, vpsi, vApsi);
                    svfloat64_t tmp = svmul_z(ptrue, vValue, vpsi);
                    vApsi = svadd_z(ptrue, tmp, vApsi);
                    svst1(ptrue, ApsiPtr_offset + br, vApsi);
                }
            }
        }
    }
}
void Foam::ellMatrix::SpMV_UNROOL32_SVE
(
    scalarField& Apsi,
    const scalarField& psi
) const
{
    assert(typeid(label) == typeid(int64_t));
    assert(svcntd() == 8);

    scalar* ApsiPtr = Apsi.begin();
    const scalar* const __restrict__ psiPtr = psi.begin();
    const scalar* const __restrict__ diagPtr = diag_value_.begin();

    const label* const __restrict__ off_diag_colidx_Ptr = off_diag_colidx_.begin();
    const scalar* const __restrict__ off_diag_value_Ptr = off_diag_value_.begin();
    svbool_t ptrue = svptrue_b64();


    #pragma omp parallel 
    {
        int thread_rank = omp_get_thread_num();
        int thread_size = omp_get_num_threads();
        label local_start = (thread_rank * block_count_) / thread_size;
        label local_end = ((thread_rank + 1) * block_count_) / thread_size;
        // #pragma omp for schedule(static, 1)
        // for(label bi = 0; bi < block_count_; ++bi){
        for(label bi = local_start; bi < local_end; ++bi){
            label rbs = ELL_BLOCK_START(bi);
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
            // for(label br = 0; br < rbl; br += svcntd()){
            //     svfloat64_t vdiag = svld1(ptrue, diagPtr_offset + br);
            //     svfloat64_t vpsi = svld1(ptrue, psiPtr + rbs + br);
            //     svfloat64_t vApsi = svmul_z(ptrue, vdiag, vpsi);
            //     svst1(ptrue, ApsiPtr_offset + br, vApsi);
            // }
            label index_block_start = ELL_INDEX_BLOCK_START(rbs);
            for(label ellcol = 0; ellcol < max_count_; ++ellcol){
                label index_ellcol_start = index_block_start + ELL_COL_OFFSET(ellcol);
                vIndex0 = svld1_vnum(ptrue, off_diag_colidx_Ptr + index_ellcol_start, 0);
                vIndex1 = svld1_vnum(ptrue, off_diag_colidx_Ptr + index_ellcol_start, 1);
                vIndex2 = svld1_vnum(ptrue, off_diag_colidx_Ptr + index_ellcol_start, 2);
                vIndex3 = svld1_vnum(ptrue, off_diag_colidx_Ptr + index_ellcol_start, 3);
                vpsi0 = svld1_gather_index(ptrue, psiPtr, vIndex0);
                vpsi1 = svld1_gather_index(ptrue, psiPtr, vIndex1);
                vpsi2 = svld1_gather_index(ptrue, psiPtr, vIndex2);
                vpsi3 = svld1_gather_index(ptrue, psiPtr, vIndex3);
                vValue0 = svld1_vnum(ptrue, off_diag_value_Ptr + index_ellcol_start, 0);
                vValue1 = svld1_vnum(ptrue, off_diag_value_Ptr + index_ellcol_start, 1);
                vValue2 = svld1_vnum(ptrue, off_diag_value_Ptr + index_ellcol_start, 2);
                vValue3 = svld1_vnum(ptrue, off_diag_value_Ptr + index_ellcol_start, 3);
                tmp0 = svmul_z(ptrue, vValue0, vpsi0);
                tmp1 = svmul_z(ptrue, vValue1, vpsi1);
                tmp2 = svmul_z(ptrue, vValue2, vpsi2);
                tmp3 = svmul_z(ptrue, vValue3, vpsi3);
                vApsi0 = svadd_z(ptrue, tmp0, vApsi0);
                vApsi1 = svadd_z(ptrue, tmp1, vApsi1);
                vApsi2 = svadd_z(ptrue, tmp2, vApsi2);
                vApsi3 = svadd_z(ptrue, tmp3, vApsi3);
                // for(label br = 0; br < rbl; br += svcntd()){
                //     svint64_t vIndex = svld1(ptrue, off_diag_colidx_Ptr + index_ellcol_start + br);
                //     svfloat64_t vpsi = svld1_gather_index(ptrue, psiPtr, vIndex);
                //     svfloat64_t vApsi = svld1(ptrue, ApsiPtr_offset + br);
                //     svfloat64_t vValue = svld1(ptrue, off_diag_value_Ptr + index_ellcol_start + br);
                //     // svfloat64_t vApsi_o = svmla_z(ptrue, vApsi, vValue, vpsi);
                //     // svfloat64_t vApsi_o = svmad_z(ptrue, vValue, vpsi, vApsi);
                //     svfloat64_t tmp = svmul_z(ptrue, vValue, vpsi);
                //     vApsi = svadd_z(ptrue, tmp, vApsi);
                //     svst1(ptrue, ApsiPtr_offset + br, vApsi);
                // }
            }
            svst1_vnum(ptrue, ApsiPtr_offset, 0, vApsi0);
            svst1_vnum(ptrue, ApsiPtr_offset, 1, vApsi1);
            svst1_vnum(ptrue, ApsiPtr_offset, 2, vApsi2);
            svst1_vnum(ptrue, ApsiPtr_offset, 3, vApsi3);
        }
    }
    
}
#endif
// ************************************************************************* //
