/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2018-2020 OpenCFD Ltd.
    Copyright (C) 2019-2020 Simone Bna
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
    Foam::amgxSolver

Group
    grpLduMatrixSolvers

Description
    AMGxWrapper solver interface for lduMatrices using a run-time selectable solver
    and preconditioner.

SourceFiles
    amgxSolver.C

\*---------------------------------------------------------------------------*/

#ifndef amgxFoamSolver_H
#define amgxFoamSolver_H

#include "lduMatrix.H"
#include "AmgXCSRMatrix.H"


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                         Class amgxSolver Declaration
\*---------------------------------------------------------------------------*/

class amgxSolver
:
    public lduMatrix::solver
{
    // Private Data

        // Coefficients for the amgx solvers
        dictionary amgxDict_;

        //- Equation name for solver context
        word eqName_;

    // Private Member Functions

        //- No copy construct
        amgxSolver(const amgxSolver&) = delete;

        //- No copy assignment
        void operator=(const amgxSolver&) = delete;


public:

    //- Runtime type information
    TypeName("amgx");


    // Constructors

        //- Construct from matrix components and solver controls
        amgxSolver
        (
            const word& fieldName,
            const lduMatrix& matrix,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const FieldField<Field, scalar>& interfaceIntCoeffs,
            const lduInterfaceFieldPtrsList& interfaces,
            const dictionary& solverControls
        );


    //- Destructor
    virtual ~amgxSolver() = default;


    // Member Functions

        //virtual solverPerformance scalarSolve
        //(
        //    solveScalarField& psi,
        //    const solveScalarField& source,
        //    const direction cmpt = 0
        //) const;

        //- Solve the matrix with this solver
        virtual solverPerformance solve
        (
            scalarField& psi,
            const scalarField& source,
            const direction cmpt = 0
        ) const;

        void offloadMatrixArrays
        (
            AmgXCSRMatrix& Amat,
            int& nRowsLocal,
            int& nRowsGlobal,
            int& nNz
        ) const;

        void offloadMatrixValues
        (
            AmgXCSRMatrix& Amat,
            int& nRowsLocal,
            int& nNz
        ) const;

};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
