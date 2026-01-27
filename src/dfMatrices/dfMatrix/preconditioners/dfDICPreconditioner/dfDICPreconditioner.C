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

\*---------------------------------------------------------------------------*/

#include "dfDICPreconditioner.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(dfDICPreconditioner, 0);

    dfMatrix::preconditioner::
        addsymMatrixConstructorToTable<dfDICPreconditioner>
        adddfDICPreconditionerSymMatrixConstructorToTable_;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::dfDICPreconditioner::dfDICPreconditioner
(
    const dfMatrix::solver& sol,
    const dictionary&
)
:
    dfMatrix::preconditioner(sol),
    rD_(sol.matrix().diag())
{
    calcReciprocalD(rD_, sol.matrix());
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::dfDICPreconditioner::calcReciprocalD
(
    scalarField& rD,
    const dfMatrix& matrix
)
{
    // scalar* __restrict__ rDPtr = rD.begin();

    // const label* const __restrict__ uPtr = matrix.lduAddr().upperAddr().begin();
    // const label* const __restrict__ lPtr = matrix.lduAddr().lowerAddr().begin();
    // const scalar* const __restrict__ upperPtr = matrix.upper().begin();

    // // Calculate the DIC diagonal
    // const label nFaces = matrix.upper().size();
    // for (label face=0; face<nFaces; face++)
    // {
    //     rDPtr[uPtr[face]] -= upperPtr[face]*upperPtr[face]/rDPtr[lPtr[face]];
    // }


    // // Calculate the reciprocal of the preconditioned diagonal
    // const label nCells = rD.size();

    // for (label cell=0; cell<nCells; cell++)
    // {
    //     rDPtr[cell] = 1.0/rDPtr[cell];
    // }
    matrix.calcDICReciprocalD(rD);
}


void Foam::dfDICPreconditioner::precondition
(
    scalarField& wA,
    const scalarField& rA,
    const direction
) const
{
    // scalar* __restrict__ wAPtr = wA.begin();
    // const scalar* __restrict__ rAPtr = rA.begin();
    // const scalar* __restrict__ rDPtr = rD_.begin();

    // const label* const __restrict__ uPtr =
    //     solver_.matrix().lduAddr().upperAddr().begin();
    // const label* const __restrict__ lPtr =
    //     solver_.matrix().lduAddr().lowerAddr().begin();
    // const scalar* const __restrict__ upperPtr =
    //     solver_.matrix().upper().begin();

    // label nCells = wA.size();
    // label nFaces = solver_.matrix().upper().size();
    // label nFacesM1 = nFaces - 1;

    // for (label cell=0; cell<nCells; cell++)
    // {
    //     wAPtr[cell] = rDPtr[cell]*rAPtr[cell];
    // }

    // for (label face=0; face<nFaces; face++)
    // {
    //     wAPtr[uPtr[face]] -= rDPtr[uPtr[face]]*upperPtr[face]*wAPtr[lPtr[face]];
    // }

    // for (label face=nFacesM1; face>=0; face--)
    // {
    //     wAPtr[lPtr[face]] -= rDPtr[lPtr[face]]*upperPtr[face]*wAPtr[uPtr[face]];
    // }
    solver_.matrix().DICPrecondition(wA, rA, rD_);
}


// ************************************************************************* //
