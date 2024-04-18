/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2011-2019 OpenFOAM Foundation
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

Class
    Foam::LDUNoPreconditioner

Description
    Null preconditioner for both symmetric and asymmetric matrices.

SourceFiles
    LDUNoPreconditioner.C

\*---------------------------------------------------------------------------*/

#ifndef LDUNoPreconditioner_H
#define LDUNoPreconditioner_H

#include "LDUMatrix.H"
#include "LDUNoPreconditioner.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                           Class LDUNoPreconditioner Declaration
\*---------------------------------------------------------------------------*/

class LDUNoPreconditioner
:
    public LDUMatrix::preconditioner
{

public:

    //- Runtime type information
    TypeName("none");


    // Constructors

        //- Construct from matrix components and preconditioner solver controls
        LDUNoPreconditioner
        (
            const LDUMatrix::solver&,
            const dictionary& solverControlsUnused
        );

        //- Disallow default bitwise copy construction
        LDUNoPreconditioner(const LDUNoPreconditioner&) = delete;


    //- Destructor
    virtual ~LDUNoPreconditioner()
    {}


    // Member Functions

        //- Return wA the preconditioned form of residual rA
        virtual void precondition
        (
            scalarField& wA,
            const scalarField& rA,
            const direction cmpt=0
        ) const;

        //- Return wT the transpose-matrix preconditioned form of residual rT.
        virtual void preconditionT
        (
            scalarField& wT,
            const scalarField& rT,
            const direction cmpt=0
        ) const
        {
            return precondition(wT, rT, cmpt);
        }


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const LDUNoPreconditioner&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
