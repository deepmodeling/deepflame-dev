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

#include "dfNoPreconditioner.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(dfNoPreconditioner, 0);

    dfMatrix::preconditioner::
        addsymMatrixConstructorToTable<dfNoPreconditioner>
        adddfNoPreconditionerSymMatrixConstructorToTable_;

    dfMatrix::preconditioner::
        addasymMatrixConstructorToTable<dfNoPreconditioner>
        adddfNoPreconditionerAsymMatrixConstructorToTable_;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::dfNoPreconditioner::dfNoPreconditioner
(
    const dfMatrix::solver& sol,
    const dictionary&
)
:
    dfMatrix::preconditioner(sol)
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::dfNoPreconditioner::precondition
(
    scalarField& wA,
    const scalarField& rA,
    const direction
) const
{
    scalar* __restrict__ wAPtr = wA.begin();
    const scalar* __restrict__ rAPtr = rA.begin();

    label nCells = wA.size();

    for (label cell=0; cell<nCells; cell++)
    {
        wAPtr[cell] = rAPtr[cell];
    }
}


// ************************************************************************* //
