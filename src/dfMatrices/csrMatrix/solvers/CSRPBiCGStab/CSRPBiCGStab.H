/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2016-2019 OpenFOAM Foundation
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
    Foam::CSRPBiCGStab

Description
    Preconditioned bi-conjugate gradient stabilized solver for asymmetric
    lduMatrices using a run-time selectable preconditioner.

    References:
    \verbatim
        Van der Vorst, H. A. (1992).
        Bi-CGSTAB: A fast and smoothly converging variant of Bi-CG
        for the solution of nonsymmetric linear systems.
        SIAM Journal on scientific and Statistical Computing, 13(2), 631-644.

        Barrett, R., Berry, M. W., Chan, T. F., Demmel, J., Donato, J.,
        Dongarra, J., Eijkhout, V., Pozo, R., Romine, C. & Van der Vorst, H.
        (1994).
        Templates for the solution of linear systems:
        building blocks for iterative methods
        (Vol. 43). Siam.
    \endverbatim

SourceFiles
    CSRPBiCGStab.C

\*---------------------------------------------------------------------------*/

#ifndef CSRPBiCGStab_H
#define CSRPBiCGStab_H

#include "csrMatrix.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                           Class CSRPBiCGStab Declaration
\*---------------------------------------------------------------------------*/

class CSRPBiCGStab
:
    public csrMatrix::solver
{
    autoPtr<csrMatrix::preconditioner> preconPtr_;

public:

    //- Runtime type information
    TypeName("PBiCGStab");


    // Constructors

        //- Construct from matrix components and solver data stream
        CSRPBiCGStab
        (
            const word& fieldName,
            const csrMatrix& matrix,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const FieldField<Field, scalar>& interfaceIntCoeffs,
            const lduInterfaceFieldPtrsList& interfaces,
            const dictionary& solverControls
        );

        //- Disallow default bitwise copy construction
        CSRPBiCGStab(const CSRPBiCGStab&) = delete;


    //- Destructor
    virtual ~CSRPBiCGStab()
    {}


    // Member Functions

        //- Solve the matrix with this solver
        virtual solverPerformance solve
        (
            scalarField& psi,
            const scalarField& source,
            const direction cmpt=0
        ) const;


    // Member Operators

        //- Disallow default bitwise assignment
        void operator=(const CSRPBiCGStab&) = delete;

    // profiling
    mutable double solve_time = 0.;
    mutable double spmv_time = 0.;
    mutable double normFactor_time = 0.;
    mutable double gSumMag_time = 0.;
    mutable double gSumProd_time = 0.;
    mutable double gSumSqr_time = 0.;
    mutable double localUpdate_time = 0.;
    mutable double precondition_time = 0.;

    void print_time() const {
        double other_time = solve_time - spmv_time - normFactor_time - gSumMag_time - gSumProd_time - gSumSqr_time - localUpdate_time - precondition_time;
        Info << "CSRPCG solve profiling -----------------------------------------------------------------" << endl;
        Info << "solve time : " << solve_time << endl;
        Info << "spmv time : " << spmv_time << ", " << spmv_time / solve_time * 100 << "%" << endl;
        Info << "normFactor time : " << normFactor_time << ", " << normFactor_time / solve_time * 100 << "%" << endl;
        Info << "gSumMag time : " << gSumMag_time << ", " << gSumMag_time / solve_time * 100 << "%" << endl;
        Info << "gSumProd time : " << gSumProd_time << ", " << gSumProd_time / solve_time * 100 << "%" << endl;
        Info << "gSumSqr time : " << gSumSqr_time << ", " << gSumSqr_time / solve_time * 100 << "%" << endl;
        Info << "localUpdate time : " << localUpdate_time << ", " << localUpdate_time / solve_time * 100 << "%" << endl;
        Info << "precondition time : " << precondition_time << ", " << precondition_time / solve_time * 100 << "%" << endl;
        Info << "other time : " << other_time << ", " << other_time / solve_time * 100 << "%" << endl;
        Info << "----------------------------------------------------------------------------------------" << endl;
        preconPtr_->print_time();
    }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
