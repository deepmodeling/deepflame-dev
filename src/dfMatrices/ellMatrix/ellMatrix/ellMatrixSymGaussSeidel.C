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

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
void Foam::ellMatrix::SymGaussSeidel
(
    scalarField& psi,
    const scalarField& bPrime
) const
{

    SymGaussSeidel_naive(psi, bPrime);
}

void Foam::ellMatrix::SymGaussSeidel_naive
(
    scalarField& psi,
    const scalarField& bPrime
) const
{
    assert(psi.size() == row_);


    const scalar* const __restrict__ bPrimePtr = bPrime.begin();

    scalar* __restrict__ psiPtr = psi.begin();

    const scalar* const __restrict__ diagPtr = diag_value_.begin();
    const label* const __restrict__ off_diag_colidx_Ptr = off_diag_colidx_.begin();
    const scalar* const __restrict__ off_diag_value_Ptr = off_diag_value_.begin();

    scalarField tmp(row_block_size_);
    scalar* __restrict__ tmpPtr = tmp.begin();

    for(label bi = 0; bi < block_count_; ++bi){
        label rbs = ELL_BLOCK_START(bi);
        label rbe = ELL_BLOCK_END(rbs);
        label rbl = ELL_BLOCK_LEN(rbs,rbe);
        scalar* __restrict__ psiPtr_offset = psiPtr + rbs;
        const scalar* const __restrict__ bPrimePtr_offset = bPrimePtr + rbs;
        const scalar* const __restrict__ diagPtr_offset = diagPtr + rbs;
        for(label br = 0; br < rbl; ++br){
            tmpPtr[br] = bPrimePtr_offset[br];
        }
        label index_block_start = ELL_INDEX_BLOCK_START(rbs);
        for(label ellcol = 0; ellcol < max_count_; ++ellcol){
            label index_ellcol_start = index_block_start + ELL_COL_OFFSET(ellcol);
            for(label br = 0; br < rbl; ++br){
                scalar value = off_diag_value_Ptr[index_ellcol_start + br];
                label col = off_diag_colidx_Ptr[index_ellcol_start + br];
                tmpPtr[br] -= value * psiPtr[col];
            }
        }
        for(label br = 0; br < rbl; ++br){
            psiPtr_offset[br] = tmpPtr[br] / diagPtr_offset[br];
        }
    }

    for(label bi = block_count_ - 1; bi >= 0; --bi){
        label rbs = ELL_BLOCK_START(bi);
        label rbe = ELL_BLOCK_END(rbs);
        label rbl = ELL_BLOCK_LEN(rbs,rbe);
        scalar* psiPtr_offset = psiPtr + rbs;
        const scalar* const __restrict__ bPrimePtr_offset = bPrimePtr + rbs;
        const scalar* const __restrict__ diagPtr_offset = diagPtr + rbs;
        for(label br = 0; br < rbl; ++br){
            tmpPtr[br] = bPrimePtr_offset[br];
        }
        label index_block_start = ELL_INDEX_BLOCK_START(rbs);
        for(label ellcol = 0; ellcol < max_count_; ++ellcol){
            label index_ellcol_start = index_block_start + ELL_COL_OFFSET(ellcol);
            for(label br = 0; br < rbl; ++br){
                scalar value = off_diag_value_Ptr[index_ellcol_start + br];
                label col = off_diag_colidx_Ptr[index_ellcol_start + br];
                tmpPtr[br] -= value * psiPtr[col];
            }
        }
        for(label br = 0; br < rbl; ++br){
            psiPtr_offset[br] = tmpPtr[br] / diagPtr_offset[br];
        }
    }
}

void Foam::ellMatrix::SymGaussSeidel_B8
(
    scalarField& psi,
    const scalarField& bPrime
) const
{
    assert(psi.size() == row_);
    assert(row_block_size_ == 8 && ELL_BLOCK_TAIL == 0);

    const scalar* const __restrict__ bPrimePtr = bPrime.begin();

    scalar* __restrict__ psiPtr = psi.begin();

    const scalar* const __restrict__ diagPtr = diag_value_.begin();
    const label* const __restrict__ off_diag_colidx_Ptr = off_diag_colidx_.begin();
    const scalar* const __restrict__ off_diag_value_Ptr = off_diag_value_.begin();

    scalarField tmp(row_block_size_);
    scalar* __restrict__ tmpPtr = tmp.begin();
    for(label bi = 0; bi < block_count_; ++bi){
        label rbs = ELL_BLOCK_START(bi);
        scalar* __restrict__ psiPtr_offset = psiPtr + rbs;
        const scalar* const __restrict__ bPrimePtr_offset = bPrimePtr + rbs;
        const scalar* const __restrict__ diagPtr_offset = diagPtr + rbs;
        tmpPtr[0] = bPrimePtr_offset[0];
        tmpPtr[1] = bPrimePtr_offset[1];
        tmpPtr[2] = bPrimePtr_offset[2];
        tmpPtr[3] = bPrimePtr_offset[3];
        tmpPtr[4] = bPrimePtr_offset[4];
        tmpPtr[5] = bPrimePtr_offset[5];
        tmpPtr[6] = bPrimePtr_offset[6];
        tmpPtr[7] = bPrimePtr_offset[7];
        label index_block_start = ELL_INDEX_BLOCK_START(rbs);
        for(label ellcol = 0; ellcol < max_count_; ++ellcol){
            label index_ellcol_start = index_block_start + ELL_COL_OFFSET(ellcol);
            tmpPtr[0] -= off_diag_value_Ptr[index_ellcol_start + 0] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + 0]];
            tmpPtr[1] -= off_diag_value_Ptr[index_ellcol_start + 1] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + 1]];
            tmpPtr[2] -= off_diag_value_Ptr[index_ellcol_start + 2] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + 2]];
            tmpPtr[3] -= off_diag_value_Ptr[index_ellcol_start + 3] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + 3]];
            tmpPtr[4] -= off_diag_value_Ptr[index_ellcol_start + 4] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + 4]];
            tmpPtr[5] -= off_diag_value_Ptr[index_ellcol_start + 5] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + 5]];
            tmpPtr[6] -= off_diag_value_Ptr[index_ellcol_start + 6] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + 6]];
            tmpPtr[7] -= off_diag_value_Ptr[index_ellcol_start + 7] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + 7]];
        }
        psiPtr_offset[0] = tmpPtr[0] / diagPtr_offset[0];
        psiPtr_offset[1] = tmpPtr[1] / diagPtr_offset[1];
        psiPtr_offset[2] = tmpPtr[2] / diagPtr_offset[2];
        psiPtr_offset[3] = tmpPtr[3] / diagPtr_offset[3];
        psiPtr_offset[4] = tmpPtr[4] / diagPtr_offset[4];
        psiPtr_offset[5] = tmpPtr[5] / diagPtr_offset[5];
        psiPtr_offset[6] = tmpPtr[6] / diagPtr_offset[6];
        psiPtr_offset[7] = tmpPtr[7] / diagPtr_offset[7];
    }
    for(label bi = block_count_ - 1; bi >= 0; --bi){
        label rbs = ELL_BLOCK_START(bi);
        scalar* psiPtr_offset = psiPtr + rbs;
        const scalar* const __restrict__ bPrimePtr_offset = bPrimePtr + rbs;
        const scalar* const __restrict__ diagPtr_offset = diagPtr + rbs;
        tmpPtr[0] = bPrimePtr_offset[0];
        tmpPtr[1] = bPrimePtr_offset[1];
        tmpPtr[2] = bPrimePtr_offset[2];
        tmpPtr[3] = bPrimePtr_offset[3];
        tmpPtr[4] = bPrimePtr_offset[4];
        tmpPtr[5] = bPrimePtr_offset[5];
        tmpPtr[6] = bPrimePtr_offset[6];
        tmpPtr[7] = bPrimePtr_offset[7];
        label index_block_start = ELL_INDEX_BLOCK_START(rbs);
        for(label ellcol = 0; ellcol < max_count_; ++ellcol){
            label index_ellcol_start = index_block_start + ELL_COL_OFFSET(ellcol);
            tmpPtr[0] -= off_diag_value_Ptr[index_ellcol_start + 0] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + 0]];
            tmpPtr[1] -= off_diag_value_Ptr[index_ellcol_start + 1] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + 1]];
            tmpPtr[2] -= off_diag_value_Ptr[index_ellcol_start + 2] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + 2]];
            tmpPtr[3] -= off_diag_value_Ptr[index_ellcol_start + 3] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + 3]];
            tmpPtr[4] -= off_diag_value_Ptr[index_ellcol_start + 4] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + 4]];
            tmpPtr[5] -= off_diag_value_Ptr[index_ellcol_start + 5] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + 5]];
            tmpPtr[6] -= off_diag_value_Ptr[index_ellcol_start + 6] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + 6]];
            tmpPtr[7] -= off_diag_value_Ptr[index_ellcol_start + 7] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + 7]];
        }
        psiPtr_offset[0] = tmpPtr[0] / diagPtr_offset[0];
        psiPtr_offset[1] = tmpPtr[1] / diagPtr_offset[1];
        psiPtr_offset[2] = tmpPtr[2] / diagPtr_offset[2];
        psiPtr_offset[3] = tmpPtr[3] / diagPtr_offset[3];
        psiPtr_offset[4] = tmpPtr[4] / diagPtr_offset[4];
        psiPtr_offset[5] = tmpPtr[5] / diagPtr_offset[5];
        psiPtr_offset[6] = tmpPtr[6] / diagPtr_offset[6];
        psiPtr_offset[7] = tmpPtr[7] / diagPtr_offset[7];
    }
}

// void Foam::ellMatrix::SymGaussSeidel_B8_colored
// (
//     scalarField& psi,
//     const scalarField& bPrime
// ) const
// {
//     assert(psi.size() == row_);
//     assert(colored_);

//     const scalar* const __restrict__ bPrimePtr = bPrime.begin();

//     scalar* __restrict__ psiPtr = psi.begin();

//     const scalar* const __restrict__ diagPtr = diag_value_.begin();
//     const label* const __restrict__ off_diag_colidx_Ptr = off_diag_colidx_.begin();
//     const scalar* const __restrict__ off_diag_value_Ptr = off_diag_value_.begin();

//     #pragma omp parallel
//     {
//         scalarField tmp(row_block_size_);
//         scalar* __restrict__ tmpPtr = tmp.begin();
//         for(label color = 0; color < num_color_; ++color){
//             #pragma omp for
//             for(label node_index = color_ptr_[color]; node_index < color_ptr_[color+1]; ++node_index){
//                 label bi = nodes_of_color_[node_index];
//                 label rbs = ELL_BLOCK_START(bi);
//                 scalar* __restrict__ psiPtr_offset = psiPtr + rbs;
//                 const scalar* const __restrict__ bPrimePtr_offset = bPrimePtr + rbs;
//                 const scalar* const __restrict__ diagPtr_offset = diagPtr + rbs;
//                 tmpPtr[0] = bPrimePtr_offset[0];
//                 tmpPtr[1] = bPrimePtr_offset[1];
//                 tmpPtr[2] = bPrimePtr_offset[2];
//                 tmpPtr[3] = bPrimePtr_offset[3];
//                 tmpPtr[4] = bPrimePtr_offset[4];
//                 tmpPtr[5] = bPrimePtr_offset[5];
//                 tmpPtr[6] = bPrimePtr_offset[6];
//                 tmpPtr[7] = bPrimePtr_offset[7];
//                 label index_block_start = ELL_INDEX_BLOCK_START(rbs);
//                 for(label ellcol = 0; ellcol < max_count_; ++ellcol){
//                     label index_ellcol_start = index_block_start + ELL_COL_OFFSET(ellcol);
//                     tmpPtr[0] -= off_diag_value_Ptr[index_ellcol_start + 0] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + 0]];
//                     tmpPtr[1] -= off_diag_value_Ptr[index_ellcol_start + 1] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + 1]];
//                     tmpPtr[2] -= off_diag_value_Ptr[index_ellcol_start + 2] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + 2]];
//                     tmpPtr[3] -= off_diag_value_Ptr[index_ellcol_start + 3] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + 3]];
//                     tmpPtr[4] -= off_diag_value_Ptr[index_ellcol_start + 4] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + 4]];
//                     tmpPtr[5] -= off_diag_value_Ptr[index_ellcol_start + 5] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + 5]];
//                     tmpPtr[6] -= off_diag_value_Ptr[index_ellcol_start + 6] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + 6]];
//                     tmpPtr[7] -= off_diag_value_Ptr[index_ellcol_start + 7] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + 7]];
//                 }
//                 psiPtr_offset[0] = tmpPtr[0] / diagPtr_offset[0];
//                 psiPtr_offset[1] = tmpPtr[1] / diagPtr_offset[1];
//                 psiPtr_offset[2] = tmpPtr[2] / diagPtr_offset[2];
//                 psiPtr_offset[3] = tmpPtr[3] / diagPtr_offset[3];
//                 psiPtr_offset[4] = tmpPtr[4] / diagPtr_offset[4];
//                 psiPtr_offset[5] = tmpPtr[5] / diagPtr_offset[5];
//                 psiPtr_offset[6] = tmpPtr[6] / diagPtr_offset[6];
//                 psiPtr_offset[7] = tmpPtr[7] / diagPtr_offset[7];
//             }
//         }
//         for(label color = num_color_ - 1; color >= 0; --color){
//             #pragma omp for
//             for(label node_index = color_ptr_[color]; node_index < color_ptr_[color+1]; ++node_index){
//                 label bi = nodes_of_color_[node_index];
//                 label rbs = ELL_BLOCK_START(bi);
//                 scalar* __restrict__ psiPtr_offset = psiPtr + rbs;
//                 const scalar* const __restrict__ bPrimePtr_offset = bPrimePtr + rbs;
//                 const scalar* const __restrict__ diagPtr_offset = diagPtr + rbs;
//                 tmpPtr[0] = bPrimePtr_offset[0];
//                 tmpPtr[1] = bPrimePtr_offset[1];
//                 tmpPtr[2] = bPrimePtr_offset[2];
//                 tmpPtr[3] = bPrimePtr_offset[3];
//                 tmpPtr[4] = bPrimePtr_offset[4];
//                 tmpPtr[5] = bPrimePtr_offset[5];
//                 tmpPtr[6] = bPrimePtr_offset[6];
//                 tmpPtr[7] = bPrimePtr_offset[7];
//                 label index_block_start = ELL_INDEX_BLOCK_START(rbs);
//                 for(label ellcol = 0; ellcol < max_count_; ++ellcol){
//                     label index_ellcol_start = index_block_start + ELL_COL_OFFSET(ellcol);
//                     tmpPtr[0] -= off_diag_value_Ptr[index_ellcol_start + 0] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + 0]];
//                     tmpPtr[1] -= off_diag_value_Ptr[index_ellcol_start + 1] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + 1]];
//                     tmpPtr[2] -= off_diag_value_Ptr[index_ellcol_start + 2] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + 2]];
//                     tmpPtr[3] -= off_diag_value_Ptr[index_ellcol_start + 3] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + 3]];
//                     tmpPtr[4] -= off_diag_value_Ptr[index_ellcol_start + 4] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + 4]];
//                     tmpPtr[5] -= off_diag_value_Ptr[index_ellcol_start + 5] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + 5]];
//                     tmpPtr[6] -= off_diag_value_Ptr[index_ellcol_start + 6] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + 6]];
//                     tmpPtr[7] -= off_diag_value_Ptr[index_ellcol_start + 7] * psiPtr[off_diag_colidx_Ptr[index_ellcol_start + 7]];
//                 }
//                 psiPtr_offset[0] = tmpPtr[0] / diagPtr_offset[0];
//                 psiPtr_offset[1] = tmpPtr[1] / diagPtr_offset[1];
//                 psiPtr_offset[2] = tmpPtr[2] / diagPtr_offset[2];
//                 psiPtr_offset[3] = tmpPtr[3] / diagPtr_offset[3];
//                 psiPtr_offset[4] = tmpPtr[4] / diagPtr_offset[4];
//                 psiPtr_offset[5] = tmpPtr[5] / diagPtr_offset[5];
//                 psiPtr_offset[6] = tmpPtr[6] / diagPtr_offset[6];
//                 psiPtr_offset[7] = tmpPtr[7] / diagPtr_offset[7];
//             }
//         }
//     }
// }

// ************************************************************************* //
