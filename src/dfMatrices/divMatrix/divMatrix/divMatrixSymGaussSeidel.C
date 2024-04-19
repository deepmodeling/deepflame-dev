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

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
void Foam::divMatrix::SymGaussSeidel
(
    scalarField& psi,
    const scalarField& bPrime
) const
{
    SymGaussSeidel_naive(psi, bPrime);
}

void Foam::divMatrix::SymGaussSeidel_naive
(
    scalarField& psi,
    const scalarField& bPrime
) const
{
    const scalar* const __restrict__ bPrimePtr = bPrime.begin();

    scalar* __restrict__ psiPtr = psi.begin();

    const scalar* const __restrict__ diagPtr = diag_value_.begin();
    const scalar* const __restrict__ off_diag_value_Ptr = off_diag_value_.begin();

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
                    psiPtr_offset[br] -= off_diag_value_Ptr[index_divcol_start + br] * psiPtr[distance + rbs + br];
                }
            }else{
                for(label br = 0; br < rbl; ++br){
                    label col = distance + rbs + br;
                    col = std::max(col, static_cast<label>(0));
                    col = std::min(col, row_ - 1);
                    psiPtr_offset[br] -= off_diag_value_Ptr[index_divcol_start + br] * psiPtr[col];
                }
            }
        }
        for(label br = 0; br < rbl; ++br){
            psiPtr_offset[br] /= diagPtr_offset[br];
        }
    }
    for(label bi = block_count_ - 1; bi >= 0; --bi){
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
                    psiPtr_offset[br] -= off_diag_value_Ptr[index_divcol_start + br] * psiPtr[distance + rbs + br];
                }
            }else{
                for(label br = 0; br < rbl; ++br){
                    label col = distance + rbs + br;
                    col = std::max(col, static_cast<label>(0));
                    col = std::min(col, row_ - 1);
                    psiPtr_offset[br] -= off_diag_value_Ptr[index_divcol_start + br] * psiPtr[col];
                }
            }
        }
        for(label br = 0; br < rbl; ++br){
            psiPtr_offset[br] /= diagPtr_offset[br];
        }
    }
}

// ************************************************************************* //
