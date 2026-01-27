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
    Foam::dfDILUPreconditioner

Description
    Simplified diagonal-based incomplete LU preconditioner for asymmetric
    matrices.  The reciprocal of the preconditioned diagonal is calculated
    and stored.

SourceFiles
    dfDILUPreconditioner.C

\*---------------------------------------------------------------------------*/

#ifndef dfDILUPreconditioner_H
#define dfDILUPreconditioner_H

#include "dfMatrix.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                           Class dfDILUPreconditioner Declaration
\*---------------------------------------------------------------------------*/

class dfDILUPreconditioner
:
    public dfMatrix::preconditioner
{
    // Private Data

        //- The reciprocal preconditioned diagonal
        scalarField rD_;


public:

    //- Runtime type information
    TypeName("DILU");


    // Constructors

        //- Construct from matrix components and preconditioner solver controls
        dfDILUPreconditioner
        (
            const dfMatrix::solver&,
            const dictionary& solverControlsUnused
        );


    //- Destructor
    virtual ~dfDILUPreconditioner()
    {}


    // Member Functions

        //- Calculate the reciprocal of the preconditioned diagonal
        static void calcReciprocalD(scalarField& rD, const dfMatrix& matrix);

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
        ) const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
