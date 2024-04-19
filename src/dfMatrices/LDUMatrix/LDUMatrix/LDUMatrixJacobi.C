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

#include "LDUMatrix.H"
#include <cassert>
#include <omp.h>
#include <mpi.h>
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h> 
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
void Foam::LDUMatrix::Jacobi
(
    scalarField& psi,
    scalarField& bPrime
) const
{
    double start = MPI_Wtime();

    Jacobi_naive(psi, bPrime);
    double end = MPI_Wtime();
    Jacobi_time_ += end - start;

}

void Foam::LDUMatrix::Jacobi_naive
(
    scalarField& psi,
    scalarField& bPrime
) const
{
    scalar* __restrict__ bPrimePtr = bPrime.begin();
    scalar* __restrict__ psiPtr = psi.begin();

    const label nCells = psi.size();
    scalarField psiCopy(nCells);
    scalar* __restrict__ psiCopyPtr = psiCopy.begin();

    const scalar* const __restrict__ diagPtr = diag().begin();
    const scalar* const __restrict__ upperPtr =
        upper().begin();
    const scalar* const __restrict__ lowerPtr =
        lower().begin();

    const label* const __restrict__ uPtr =
        lduAddr().upperAddr().begin();

    const label* const __restrict__ ownStartPtr =
        lduAddr().ownerStartAddr().begin();

    scalar psii;
    label fStart;
    label fEnd = ownStartPtr[0];

    for(label celli=0; celli<nCells; celli++){
        psiCopyPtr[celli] = psiPtr[celli];
    }

    for (label celli=0; celli<nCells; celli++)
    {
        // Start and end of this row
        fStart = fEnd;
        fEnd = ownStartPtr[celli + 1];

        // Get the accumulated neighbour side
        psii = bPrimePtr[celli];

        // Accumulate the owner product side
        for (label facei=fStart; facei<fEnd; facei++)
        {
            psii -= upperPtr[facei]*psiCopyPtr[uPtr[facei]];
        }

        // Finish psi for this cell
        psii /= diagPtr[celli];

        // Distribute the neighbour side using psi for this cell
        for (label facei=fStart; facei<fEnd; facei++)
        {
            bPrimePtr[uPtr[facei]] -= lowerPtr[facei]*psii;
        }

        psiPtr[celli] = psii;
    }
}

// ************************************************************************* //
