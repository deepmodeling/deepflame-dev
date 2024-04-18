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

#include "csrMatrix.H"
#include <cassert>
#include <typeinfo>
#include <omp.h>
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h> 
#endif
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void Foam::csrMatrix::SpMV
(
    scalarField& Apsi,
    const scalarField& psi
) const
{
    SpMV_naive(Apsi, psi);
}

void Foam::csrMatrix::SpMV_naive
(
    scalarField& Apsi,
    const scalarField& psi
) const
{
    scalar* __restrict__ ApsiPtr = Apsi.begin();
    const scalar* const __restrict__ psiPtr = psi.begin();
    const scalar* const __restrict__ diagPtr = diag_value_.begin();
    const label* const __restrict__ off_diag_rowptr_Ptr = off_diag_rowptr_.begin();
    const label* const __restrict__ off_diag_colidx_Ptr = off_diag_colidx_.begin();
    const scalar* const __restrict__ off_diag_value_Ptr = off_diag_value_.begin();

    #pragma omp parallel for
    for (label r = 0; r < row_; ++r){
        scalar tmp = diagPtr[r] * psiPtr[r];
        for(label index = off_diag_rowptr_Ptr[r]; index < off_diag_rowptr_Ptr[r+1]; ++index){
            tmp += off_diag_value_Ptr[index] * psiPtr[off_diag_colidx_Ptr[index]];
        }
        ApsiPtr[r] = tmp;
    }
}

void Foam::csrMatrix::SpMV_UNROOL
(
    scalarField& Apsi,
    const scalarField& psi
) const
{
    scalar* __restrict__ ApsiPtr = Apsi.begin();
    const scalar* const __restrict__ psiPtr = psi.begin();
    const scalar* const __restrict__ diagPtr = diag_value_.begin();
    const label* const __restrict__ off_diag_rowptr_Ptr = off_diag_rowptr_.begin();
    const label* const __restrict__ off_diag_colidx_Ptr = off_diag_colidx_.begin();
    const scalar* const __restrict__ off_diag_value_Ptr = off_diag_value_.begin();

    #pragma omp parallel for
    for (label r = 0; r < row_; ++r){
        scalar tmp = diagPtr[r] * psiPtr[r];
        const label row_len = off_diag_rowptr_Ptr[r+1] - off_diag_rowptr_Ptr[r];
        const label row_start = off_diag_rowptr_Ptr[r];
        const label* const __restrict__ off_diag_colidx_Ptr_row = &off_diag_colidx_Ptr[row_start];
        const scalar* const __restrict__ off_diag_value_Ptr_row = &off_diag_value_Ptr[row_start];
        if(row_len == 6){
            scalar tmp0 = off_diag_value_Ptr_row[0] * psiPtr[off_diag_colidx_Ptr_row[0]];
            scalar tmp1 = off_diag_value_Ptr_row[1] * psiPtr[off_diag_colidx_Ptr_row[1]];
            scalar tmp2 = off_diag_value_Ptr_row[2] * psiPtr[off_diag_colidx_Ptr_row[2]];
            scalar tmp3 = off_diag_value_Ptr_row[3] * psiPtr[off_diag_colidx_Ptr_row[3]];
            scalar tmp4 = off_diag_value_Ptr_row[4] * psiPtr[off_diag_colidx_Ptr_row[4]];
            scalar tmp5 = off_diag_value_Ptr_row[5] * psiPtr[off_diag_colidx_Ptr_row[5]];
            ApsiPtr[r] = tmp + tmp0 + tmp1 + tmp2 + tmp3 + tmp4 + tmp5;
        }else if(row_len == 5){
            scalar tmp0 = off_diag_value_Ptr_row[0] * psiPtr[off_diag_colidx_Ptr_row[0]];
            scalar tmp1 = off_diag_value_Ptr_row[1] * psiPtr[off_diag_colidx_Ptr_row[1]];
            scalar tmp2 = off_diag_value_Ptr_row[2] * psiPtr[off_diag_colidx_Ptr_row[2]];
            scalar tmp3 = off_diag_value_Ptr_row[3] * psiPtr[off_diag_colidx_Ptr_row[3]];
            scalar tmp4 = off_diag_value_Ptr_row[4] * psiPtr[off_diag_colidx_Ptr_row[4]];
            ApsiPtr[r] = tmp + tmp0 + tmp1 + tmp2 + tmp3 + tmp4;
        }else if(row_len == 4){
            scalar tmp0 = off_diag_value_Ptr_row[0] * psiPtr[off_diag_colidx_Ptr_row[0]];
            scalar tmp1 = off_diag_value_Ptr_row[1] * psiPtr[off_diag_colidx_Ptr_row[1]];
            scalar tmp2 = off_diag_value_Ptr_row[2] * psiPtr[off_diag_colidx_Ptr_row[2]];
            scalar tmp3 = off_diag_value_Ptr_row[3] * psiPtr[off_diag_colidx_Ptr_row[3]];
            ApsiPtr[r] = tmp + tmp0 + tmp1 + tmp2 + tmp3;
        }else if(row_len == 3){
            scalar tmp0 = off_diag_value_Ptr_row[0] * psiPtr[off_diag_colidx_Ptr_row[0]];
            scalar tmp1 = off_diag_value_Ptr_row[1] * psiPtr[off_diag_colidx_Ptr_row[1]];
            scalar tmp2 = off_diag_value_Ptr_row[2] * psiPtr[off_diag_colidx_Ptr_row[2]];
            ApsiPtr[r] = tmp + tmp0 + tmp1 + tmp2;
        }else{
            scalar tmp = diagPtr[r] * psiPtr[r];
            for(label index = off_diag_rowptr_Ptr[r]; index < off_diag_rowptr_Ptr[r+1]; ++index){
                tmp += off_diag_value_Ptr[index] * psiPtr[off_diag_colidx_Ptr[index]];
            }
            ApsiPtr[r] = tmp;
        }
    }
}

// ************************************************************************* //
