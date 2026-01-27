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

#include "dfPBiCGStab.H"
#include <mpi.h>
#include "dfArrayOp.H"
#include "PstreamReduceOps.H"
#include "clockTime.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(dfPBiCGStab, 0);

    dfMatrix::solver::addsymMatrixConstructorToTable<dfPBiCGStab>
        adddfPBiCGStabSymMatrixConstructorToTable_;

    dfMatrix::solver::addasymMatrixConstructorToTable<dfPBiCGStab>
        adddfPBiCGStabAsymMatrixConstructorToTable_;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::dfPBiCGStab::dfPBiCGStab
(
    const word& fieldName,
    const dfMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
:
    dfMatrix::solver
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces,
        solverControls
    ),
    preconPtr_(
        dfMatrix::preconditioner::New
        (
            *this,
            controlDict_
        )
    )
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::solverPerformance Foam::dfPBiCGStab::solve
(
    scalarField& psi,
    const scalarField& source,
    const direction cmpt
) const
{
    clockTime solveClock;

    // --- Setup class containing solver performance data
    solverPerformance solverPerf
    (
        dfMatrix::preconditioner::getName(controlDict_) + typeName,
        fieldName_
    );

    const label nCells = psi.size();

    scalar* __restrict__ psiPtr = psi.begin();

    scalarField pA(nCells);
    scalar* __restrict__ pAPtr = pA.begin();

    scalarField yA(nCells);
    scalar* __restrict__ yAPtr = yA.begin();

    scalarField rA(nCells);
    scalar* __restrict__ rAPtr = rA.begin();

    const scalar* __restrict__ const sourcePtr = source.begin();

    misc_time += solveClock.timeIncrement();

    // --- Calculate A.psi
    matrix_.Amul(yA, psi, interfaceBouCoeffs_, interfaces_, cmpt);

    spmv_time += solveClock.timeIncrement();

    // --- Calculate initial residual field
    // scalarField rA(source - yA);
    #pragma omp parallel for
    for (label cell=0; cell<nCells; cell++)
    {
        rAPtr[cell] = sourcePtr[cell] - yAPtr[cell];
    }

    localUpdate_time += solveClock.timeIncrement();

    // --- Calculate normalisation factor
    const scalar normFactor = this->normFactor(psi, source, yA, pA);

    normFactor_time += solveClock.timeIncrement();

    // --- Calculate normalised residual norm
    // solverPerf.initialResidual() = gSumMag(rA, matrix().mesh().comm()) / normFactor;
    
    scalar rASumMag = dfSumMag(rA.begin(), rA.size());

    gSumMag_time += solveClock.timeIncrement();
    
    reduce(rASumMag, sumOp<scalar>());

    allreduce_time += solveClock.timeIncrement();

    solverPerf.initialResidual() =  rASumMag / normFactor;

    solverPerf.finalResidual() = solverPerf.initialResidual();

    // --- Check convergence, solve if not converged
    if
    (
        minIter_ > 0
     || !solverPerf.checkConvergence(tolerance_, relTol_)
    )
    {
        misc_time += solveClock.timeIncrement();

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
            misc_time += solveClock.timeIncrement();

            // --- Store previous rA0rA
            const scalar rA0rAold = rA0rA;

            // rA0rA = gSumProd(rA0, rA, matrix().mesh().comm());

            rA0rA = dfSumProd(rA0.begin(), rA.begin(), rA0.size());

            gSumProd_time += solveClock.timeIncrement();

            reduce(rA0rA, sumOp<scalar>());

            allreduce_time += solveClock.timeIncrement();

            // --- Test for singularity
            if (solverPerf.checkSingularity(mag(rA0rA)))
            {
                break;
            }

            misc_time += solveClock.timeIncrement();

            // --- Update pA
            if (solverPerf.nIterations() == 0)
            {
                #pragma omp parallel for
                for (label cell=0; cell<nCells; cell++)
                {
                    pAPtr[cell] = rAPtr[cell];
                }
                localUpdate_time += solveClock.timeIncrement();
            }
            else
            {
                // --- Test for singularity
                if (solverPerf.checkSingularity(mag(omega)))
                {
                    break;
                }

                const scalar beta = (rA0rA/rA0rAold)*(alpha/omega);
                #pragma omp parallel for
                for (label cell=0; cell<nCells; cell++)
                {
                    pAPtr[cell] =
                        rAPtr[cell] + beta*(pAPtr[cell] - omega*AyAPtr[cell]);
                }
                localUpdate_time += solveClock.timeIncrement();
            }

            // --- Precondition pA
            preconPtr_->precondition(yA, pA, cmpt);
            precondition_time += solveClock.timeIncrement();
            
            // --- Calculate AyA
            matrix_.Amul(AyA, yA, interfaceBouCoeffs_, interfaces_, cmpt);
            spmv_time += solveClock.timeIncrement();

            // const scalar rA0AyA = gSumProd(rA0, AyA, matrix().mesh().comm());

            scalar rA0AyA = dfSumProd(rA0.begin(), AyA.begin(), rA0.size());

            gSumProd_time += solveClock.timeIncrement();

            reduce(rA0AyA, sumOp<scalar>());

            allreduce_time += solveClock.timeIncrement(); 

            // --- Calculate sA
            alpha = rA0rA/rA0AyA;
            // --- Calculate sA
            #pragma omp parallel for
            for (label cell=0; cell<nCells; cell++)
            {
                sAPtr[cell] = rAPtr[cell] - alpha*AyAPtr[cell];
            }
            localUpdate_time += solveClock.timeIncrement();

            // --- Test sA for convergence
            // solverPerf.finalResidual() = gSumMag(sA, matrix().mesh().comm())/normFactor;

            scalar sASumMag = dfSumMag(sA.begin(), sA.size());

            gSumMag_time += solveClock.timeIncrement();

            reduce(sASumMag, sumOp<scalar>());

            allreduce_time += solveClock.timeIncrement();

            solverPerf.finalResidual() = sASumMag / normFactor;

            if (solverPerf.checkConvergence(tolerance_, relTol_))
            {
                #pragma omp parallel for
                for (label cell=0; cell<nCells; cell++)
                {
                    psiPtr[cell] += alpha*yAPtr[cell];
                }
                solverPerf.nIterations()++;
                localUpdate_time += solveClock.timeIncrement();

                break;
            }

            // --- Precondition sA
            preconPtr_->precondition(zA, sA, cmpt);
            precondition_time += solveClock.timeIncrement();

            // --- Calculate tA
            matrix_.Amul(tA, zA, interfaceBouCoeffs_, interfaces_, cmpt);
            spmv_time += solveClock.timeIncrement();

            // const scalar tAtA = gSumSqr(tA, matrix().mesh().comm());
            scalar tAtA = dfSumSqr(tA.begin(), tA.size());

            gSumSqr_time += solveClock.timeIncrement();

            reduce(tAtA, sumOp<scalar>());

            // --- Calculate omega from tA and sA
            //     (cheaper than using zA with preconditioned tA)
            // omega = gSumProd(tA, sA, matrix().mesh().comm())/tAtA;

            omega = dfSumProd(tA.begin(), sA.begin(), tA.size());

            gSumProd_time += solveClock.timeIncrement();

            reduce(omega, sumOp<scalar>());

            allreduce_time += solveClock.timeIncrement();

            omega = omega / tAtA;

            // --- Update solution and residual
            #pragma omp parallel for
            for (label cell=0; cell<nCells; cell++)
            {
                psiPtr[cell] += alpha*yAPtr[cell] + omega*zAPtr[cell];
                rAPtr[cell] = sAPtr[cell] - omega*tAPtr[cell];
            }
            localUpdate_time += solveClock.timeIncrement();

            // solverPerf.finalResidual() = gSumMag(rA, matrix().mesh().comm()) / normFactor;

            rASumMag = dfSumMag(rA.begin(), rA.size());

            gSumMag_time += solveClock.timeIncrement();

            reduce(rASumMag, sumOp<scalar>());

            allreduce_time += solveClock.timeIncrement();

            solverPerf.finalResidual() = rASumMag / normFactor;

        } while
        (
            (
              ++solverPerf.nIterations() < maxIter_
            && !solverPerf.checkConvergence(tolerance_, relTol_)
            )
         || solverPerf.nIterations() < minIter_
        );
    }
    misc_time += solveClock.timeIncrement();

    solve_time = solveClock.elapsedTime();

    print_time();

    return solverPerf;
}


// ************************************************************************* //
