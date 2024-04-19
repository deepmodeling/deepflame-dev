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

#include "LDUMatrix.H"
#include "LDUNoPreconditioner.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineRunTimeSelectionTable(LDUMatrix::preconditioner, symMatrix);
    defineRunTimeSelectionTable(LDUMatrix::preconditioner, asymMatrix);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

Foam::word Foam::LDUMatrix::preconditioner::getName
(
    const dictionary& solverControls
)
{
    word name;

    // handle primitive or dictionary entry
    const entry& e = solverControls.lookupEntry("preconditioner", false, false);
    if (e.isDict())
    {
        e.dict().lookup("preconditioner") >> name;
    }
    else
    {
        e.stream() >> name;
    }

    return name;
}


Foam::autoPtr<Foam::LDUMatrix::preconditioner>
Foam::LDUMatrix::preconditioner::New
(
    const solver& sol,
    const dictionary& solverControls
)
{
    word name;

    // handle primitive or dictionary entry
    const entry& e = solverControls.lookupEntry("preconditioner", false, false);
    if (e.isDict())
    {
        e.dict().lookup("preconditioner") >> name;
    }
    else
    {
        e.stream() >> name;
    }

    const dictionary& controls = e.isDict() ? e.dict() : dictionary::null;

    if (sol.matrix().symmetric())
    {
        symMatrixConstructorTable::iterator constructorIter =
            symMatrixConstructorTablePtr_->find(name);

        if (constructorIter == symMatrixConstructorTablePtr_->end())
        {
            FatalIOErrorInFunction
            (
                controls
            )   << "Unknown symmetric matrix preconditioner "
                << name << nl << nl
                << "Valid symmetric matrix preconditioners :" << endl
                << symMatrixConstructorTablePtr_->sortedToc()
                << exit(FatalIOError);
        }

        return autoPtr<LDUMatrix::preconditioner>
        (
            constructorIter()
            (
                sol,
                controls
            )
        );
        // return autoPtr<LDUMatrix::preconditioner>
        // (
        //     new LDUNoPreconditioner
        //     (
        //         sol,
        //         controls
        //     )
        // );
    }
    else if (sol.matrix().asymmetric())
    {
        asymMatrixConstructorTable::iterator constructorIter =
            asymMatrixConstructorTablePtr_->find(name);

        if (constructorIter == asymMatrixConstructorTablePtr_->end())
        {
            FatalIOErrorInFunction
            (
                controls
            )   << "Unknown asymmetric matrix preconditioner "
                << name << nl << nl
                << "Valid asymmetric matrix preconditioners :" << endl
                << asymMatrixConstructorTablePtr_->sortedToc()
                << exit(FatalIOError);
        }

        return autoPtr<LDUMatrix::preconditioner>
        (
            constructorIter()
            (
                sol,
                controls
            )
        );
        // return autoPtr<LDUMatrix::preconditioner>
        // (
        //     new LDUNoPreconditioner
        //     (
        //         sol,
        //         controls
        //     )
        // );
    }
    else
    {
        FatalIOErrorInFunction
        (
            controls
        )   << "cannot solve incomplete matrix, "
               "no diagonal or off-diagonal coefficient"
            << exit(FatalIOError);

        return autoPtr<LDUMatrix::preconditioner>(nullptr);
    }
}


// ************************************************************************* //
