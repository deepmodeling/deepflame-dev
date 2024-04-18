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
    Foam::ellDiagonalSolver

Description
    Foam::ellDiagonalSolver

SourceFiles
    ellDiagonalSolver.C

\*---------------------------------------------------------------------------*/

#ifndef ellDiagonalSolver_H
#define ellDiagonalSolver_H

#include "ellMatrix.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                           Class ellDiagonalSolver Declaration
\*---------------------------------------------------------------------------*/

class ellDiagonalSolver
:
    public ellMatrix::solver
{

public:

    //- Runtime type information
    TypeName("diagonal");


    // Constructors

        //- Construct from matrix and solver controls
        ellDiagonalSolver
        (
            const word& fieldName,
            const ellMatrix& matrix,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const FieldField<Field, scalar>& interfaceIntCoeffs,
            const lduInterfaceFieldPtrsList& interfaces,
            const dictionary& solverControls
        );

        //- Disallow default bitwise copy construction
        ellDiagonalSolver(const ellDiagonalSolver&) = delete;


    // Member Functions

        //- Read and reset the solver parameters from the given stream
        void read(const dictionary&)
        {}

        //- Solve the matrix with this solver
        solverPerformance solve
        (
            scalarField& psi,
            const scalarField& source,
            const direction cmpt=0
        ) const;


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const ellDiagonalSolver&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
