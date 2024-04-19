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
#include <typeinfo>
#include <omp.h>
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h> 
#endif
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void Foam::LDUMatrix::SpMV
(
    scalarField& Apsi,
    const scalarField& psi
) const
{
    SpMV_naive(Apsi, psi);
}

void Foam::LDUMatrix::SpMV_naive
(
    scalarField& Apsi,
    const scalarField& psi
) const
{
    scalar* ApsiPtr = Apsi.begin();
    const scalar* const __restrict__ psiPtr = psi.begin();
    const scalar* const __restrict__ diagPtr = diag().begin();

    const label* const __restrict__ uPtr = lduAddr().upperAddr().begin();
    const label* const __restrict__ lPtr = lduAddr().lowerAddr().begin();

    const scalar* const __restrict__ upperPtr = upper().begin();
    const scalar* const __restrict__ lowerPtr = lower().begin();

    const label nCells = diag().size();
    for (label cell=0; cell<nCells; cell++)
    {
        ApsiPtr[cell] = diagPtr[cell]*psiPtr[cell];
    }


    const label nFaces = upper().size();

    for (label face=0; face<nFaces; face++)
    {
        ApsiPtr[uPtr[face]] += lowerPtr[face]*psiPtr[lPtr[face]];
        ApsiPtr[lPtr[face]] += upperPtr[face]*psiPtr[uPtr[face]];
    }
}

// ************************************************************************* //
