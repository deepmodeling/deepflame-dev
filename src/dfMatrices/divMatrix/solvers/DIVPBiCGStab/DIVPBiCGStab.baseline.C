/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2016-2018 OpenFOAM Foundation
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

#include "DIVPBiCGStab.H"
#include <mpi.h>
#include <clockTime.H>

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(DIVPBiCGStab, 0);

    divMatrix::solver::addsymMatrixConstructorToTable<DIVPBiCGStab>
        addDIVPBiCGStabSymMatrixConstructorToTable_;

    divMatrix::solver::addasymMatrixConstructorToTable<DIVPBiCGStab>
        addDIVPBiCGStabAsymMatrixConstructorToTable_;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::DIVPBiCGStab::DIVPBiCGStab
(
    const word& fieldName,
    const divMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
:
    divMatrix::solver
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces,
        solverControls
    ),
    preconPtr_(
        divMatrix::preconditioner::New
        (
            *this,
            controlDict_
        )
    )
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::solverPerformance Foam::DIVPBiCGStab::solve
(
    scalarField& psi,
    const scalarField& source,
    const direction cmpt
) const
{
    clockTime clock;

    // --- Setup class containing solver performance data
    solverPerformance solverPerf
    (
        divMatrix::preconditioner::getName(controlDict_) + typeName,
        fieldName_
    );

    const label nCells = psi.size();

    scalar* __restrict__ psiPtr = psi.begin();

    scalarField pA(nCells);
    scalar* __restrict__ pAPtr = pA.begin();

    scalarField yA(nCells);
    scalar* __restrict__ yAPtr = yA.begin();

    // --- Calculate A.psi
    matrix_.Amul(yA, psi, interfaceBouCoeffs_, interfaces_, cmpt);
    
    PBiCGStab_spmv_time += clock.timeIncrement();

    // --- Calculate initial residual field
    scalarField rA(source - yA);
    PBiCGStab_localUpdate_time += clock.timeIncrement();

    scalar* __restrict__ rAPtr = rA.begin();

    // --- Calculate normalisation factor
    const scalar normFactor = this->normFactor(psi, source, yA, pA);
    PBiCGStab_normFactor_time += clock.timeIncrement();

    // --- Calculate normalised residual norm
    solverPerf.initialResidual() = gSumMag(rA, matrix().mesh().comm()) / normFactor;
    PBiCGStab_gSumMag_time += clock.timeIncrement();

    solverPerf.finalResidual() = solverPerf.initialResidual();

    // --- Check convergence, solve if not converged
    if
    (
        minIter_ > 0
     || !solverPerf.checkConvergence(tolerance_, relTol_)
    )
    {
        PBiCGStab_misc_time += clock.timeIncrement();

        scalarField AyA(nCells);
        scalar* __restrict__ AyAPtr = AyA.begin();

        scalarField sA(nCells);
        scalar* __restrict__ sAPtr = sA.begin();

        scalarField zA(nCells);
        scalar* __restrict__ zAPtr = zA.begin();

        scalarField tA(nCells);
        scalar* __restrict__ tAPtr = tA.begin();

        // --- Store initial residual
        const scalarField rA0(rA);

        // --- Initial values not used
        scalar rA0rA = 0;
        scalar alpha = 0;
        scalar omega = 0;

        // --- Solver iteration
        do
        {
            // --- Store previous rA0rA
            const scalar rA0rAold = rA0rA;

            PBiCGStab_misc_time += clock.timeIncrement();

            rA0rA = gSumProd(rA0, rA, matrix().mesh().comm());

            PBiCGStab_gSumProd_time += clock.timeIncrement();

            // --- Test for singularity
            if (solverPerf.checkSingularity(mag(rA0rA)))
            {
                break;
            }

            // --- Update pA
            if (solverPerf.nIterations() == 0)
            {
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for (label cell=0; cell<nCells; cell++)
                {
                    pAPtr[cell] = rAPtr[cell];
                }
                PBiCGStab_localUpdate_time += clock.timeIncrement();
            }
            else
            {
                // --- Test for singularity
                if (solverPerf.checkSingularity(mag(omega)))
                {
                    break;
                }

                PBiCGStab_misc_time += clock.timeIncrement();

                const scalar beta = (rA0rA/rA0rAold)*(alpha/omega);
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for (label cell=0; cell<nCells; cell++)
                {
                    pAPtr[cell] = rAPtr[cell] + beta*(pAPtr[cell] - omega*AyAPtr[cell]);
                }

                PBiCGStab_localUpdate_time += clock.timeIncrement();
            }

            // --- Precondition pA
            preconPtr_->precondition(yA, pA, cmpt);
            PBiCGStab_precondition_time += clock.timeIncrement();
            
            // --- Calculate AyA
            matrix_.Amul(AyA, yA, interfaceBouCoeffs_, interfaces_, cmpt);
            PBiCGStab_spmv_time += clock.timeIncrement();

            const scalar rA0AyA = gSumProd(rA0, AyA, matrix().mesh().comm());
            PBiCGStab_gSumProd_time += clock.timeIncrement();

            // --- Calculate sA
            alpha = rA0rA/rA0AyA;
            // --- Calculate sA
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (label cell=0; cell<nCells; cell++)
            {
                sAPtr[cell] = rAPtr[cell] - alpha*AyAPtr[cell];
            }
            PBiCGStab_localUpdate_time += clock.timeIncrement();

            // --- Test sA for convergence
            solverPerf.finalResidual() = gSumMag(sA, matrix().mesh().comm()) / normFactor;
            PBiCGStab_gSumMag_time += clock.timeIncrement();

            if (solverPerf.checkConvergence(tolerance_, relTol_))
            {
                PBiCGStab_misc_time += clock.timeIncrement();
                for (label cell=0; cell<nCells; cell++)
                {
                    psiPtr[cell] += alpha*yAPtr[cell];
                }
                solverPerf.nIterations()++;
                PBiCGStab_localUpdate_time += clock.timeIncrement();
                break;
            }

            // --- Precondition sA
            preconPtr_->precondition(zA, sA, cmpt);
            PBiCGStab_precondition_time += clock.timeIncrement();

            // --- Calculate tA
            matrix_.Amul(tA, zA, interfaceBouCoeffs_, interfaces_, cmpt);
            PBiCGStab_spmv_time += clock.timeIncrement();

            const scalar tAtA = gSumSqr(tA, matrix().mesh().comm());
            
            PBiCGStab_gSumSqr_time += clock.timeIncrement();

            // --- Calculate omega from tA and sA
            //     (cheaper than using zA with preconditioned tA)
            omega = gSumProd(tA, sA, matrix().mesh().comm())/tAtA;
            
            PBiCGStab_gSumProd_time += clock.timeIncrement();

            // --- Update solution and residual
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (label cell=0; cell<nCells; cell++)
            {
                psiPtr[cell] += alpha*yAPtr[cell] + omega*zAPtr[cell];
                rAPtr[cell] = sAPtr[cell] - omega*tAPtr[cell];
            }
            PBiCGStab_localUpdate_time += clock.timeIncrement();

            solverPerf.finalResidual() = gSumMag(rA, matrix().mesh().comm()) / normFactor;
            
            PBiCGStab_gSumMag_time += clock.timeIncrement();
        } while
        (
            (
              ++solverPerf.nIterations() < maxIter_
            && !solverPerf.checkConvergence(tolerance_, relTol_)
            )
         || solverPerf.nIterations() < minIter_
        );
    }

    PBiCGStab_time += clock.elapsedTime();
    return solverPerf;
}


// ************************************************************************* //
