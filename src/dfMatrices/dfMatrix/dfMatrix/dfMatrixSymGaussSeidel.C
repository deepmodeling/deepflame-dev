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

#include "dfMatrix.H"
#include <cassert>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
void Foam::dfMatrix::SymGaussSeidel
(
    scalarField& psi,
    scalarField& bPrime
) const
{
    SymGaussSeidel_naive(psi, bPrime);
}

void Foam::dfMatrix::SymGaussSeidel_naive
(
    scalarField& psi,
    scalarField& bPrime
) const
{
    scalar* __restrict__ psiPtr = psi.begin();

    const label nCells = psi.size();

    scalar* __restrict__ bPrimePtr = bPrime.begin();

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
            psii -= upperPtr[facei]*psiPtr[uPtr[facei]];
        }

        // Finish current psi
        psii /= diagPtr[celli];

        // Distribute the neighbour side using current psi
        for (label facei=fStart; facei<fEnd; facei++)
        {
            bPrimePtr[uPtr[facei]] -= lowerPtr[facei]*psii;
        }

        psiPtr[celli] = psii;
    }

    fStart = ownStartPtr[nCells];

    for (label celli=nCells-1; celli>=0; celli--)
    {
        // Start and end of this row
        fEnd = fStart;
        fStart = ownStartPtr[celli];

        // Get the accumulated neighbour side
        psii = bPrimePtr[celli];

        // Accumulate the owner product side
        for (label facei=fStart; facei<fEnd; facei++)
        {
            psii -= upperPtr[facei]*psiPtr[uPtr[facei]];
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
