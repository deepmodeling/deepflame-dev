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

    label gNCells = nCells;
    reduce(gNCells, sumOp<label>());
    PBiCGStab_allreduce_time += clock.timeIncrement();

    const scalar* __restrict__ sourcePtr = source.begin();

    scalar* __restrict__ psiPtr = psi.begin();

    scalarField pA(nCells);
    scalar* __restrict__ pAPtr = pA.begin();

    scalarField yA(nCells);
    scalar* __restrict__ yAPtr = yA.begin();

    scalarField rA(nCells);
    scalar* __restrict__ rAPtr = rA.begin();

    PBiCGStab_misc_time += clock.timeIncrement();

    // --- Calculate A.psi
    matrix_.Amul(yA, psi, interfaceBouCoeffs_, interfaces_, cmpt);
    
    PBiCGStab_spmv_time += clock.timeIncrement();

    // --- Calculate initial residual field
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(label c = 0; c < nCells; ++c){
        rAPtr[c] = sourcePtr[c] - yAPtr[c];
    }

    PBiCGStab_axpy_time += clock.timeIncrement();

    // --- Calculate normalisation factor
    matrix_.sumA(pA, interfaceBouCoeffs_, interfaces_);

    PBiCGStab_sumA_time += clock.timeIncrement();

    scalar gPsiSum = 0.;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:gPsiSum)
#endif
    for(label c = 0; c < nCells; ++c){
        gPsiSum += psi[c];
    }

    PBiCGStab_reduce_local_time += clock.timeIncrement();

    reduce(gPsiSum, sumOp<scalar>());

    PBiCGStab_allreduce_time += clock.timeIncrement();

    scalar gPsiAvg = gPsiSum / gNCells;

    scalar gTmpSum = 0.;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:gTmpSum)
#endif
    for(label c = 0; c < nCells; ++c){
        pAPtr[c] *= gPsiAvg;
        gTmpSum += std::abs(yAPtr[c] - pAPtr[c]) + std::abs(sourcePtr[c] - pAPtr[c]);
    }

    PBiCGStab_norm_local_time += clock.timeIncrement();

    reduce(gTmpSum, sumOp<scalar>());

    PBiCGStab_allreduce_time += clock.timeIncrement();
    
    scalar normFactor = gTmpSum + solverPerformance::small_;

    // --- Calculate normalised residual norm
    scalar gRASumMag = 0.;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:gRASumMag)
#endif
    for(label c = 0; c < nCells; ++c){
        gRASumMag += std::abs(rAPtr[c]);
    }

    PBiCGStab_reduce_local_time += clock.timeIncrement();

    reduce(gRASumMag, sumOp<scalar>());

    PBiCGStab_allreduce_time += clock.timeIncrement();
    
    solverPerf.initialResidual() = gRASumMag / normFactor;
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
        const scalar* __restrict__ rA0Ptr = rA0.begin();

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

            rA0rA = 0.;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:rA0rA)
#endif
            for(label c = 0; c < nCells; ++c){
                rA0rA += rA0Ptr[c] * rAPtr[c];
            }

            PBiCGStab_reduce_local_time += clock.timeIncrement();

            reduce(rA0rA, sumOp<scalar>());

            PBiCGStab_allreduce_time += clock.timeIncrement();

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
                PBiCGStab_axpy_time += clock.timeIncrement();
            }
            else
            {
                const scalar beta = (rA0rA/rA0rAold)*(alpha/omega);
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for (label cell=0; cell<nCells; cell++)
                {
                    pAPtr[cell] = rAPtr[cell] + beta * (pAPtr[cell] - omega * AyAPtr[cell]);
                }

                PBiCGStab_axpy_time += clock.timeIncrement();
            }

            // --- Precondition pA
            preconPtr_->precondition(yA, pA, cmpt);

            PBiCGStab_precondition_time += clock.timeIncrement();
            
            // --- Calculate AyA
            matrix_.Amul(AyA, yA, interfaceBouCoeffs_, interfaces_, cmpt);
            PBiCGStab_spmv_time += clock.timeIncrement();

            scalar rA0AyA = 0.;

#ifdef _OPENMP
#pragma omp parallel for reduction(+:rA0AyA)
#endif
            for(label c = 0; c < nCells; ++c){
                rA0AyA += rA0Ptr[c] * AyAPtr[c];
            }

            PBiCGStab_reduce_local_time += clock.timeIncrement();

            reduce(rA0AyA, sumOp<scalar>());

            PBiCGStab_allreduce_time += clock.timeIncrement();

            // --- Calculate sA
            alpha = rA0rA/rA0AyA;
            // --- Calculate sA
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (label cell=0; cell<nCells; cell++)
            {
                sAPtr[cell] = rAPtr[cell] - alpha * AyAPtr[cell];
            }

            PBiCGStab_axpy_time += clock.timeIncrement();

            // --- Test sA for convergence
            scalar gSASumMag = 0.;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:gSASumMag)
#endif
            for(label c = 0; c < nCells; ++c){
                gSASumMag += std::abs(sAPtr[c]);
            }

            PBiCGStab_reduce_local_time += clock.timeIncrement();

            reduce(gSASumMag, sumOp<scalar>());

            PBiCGStab_allreduce_time += clock.timeIncrement();

            solverPerf.finalResidual() = gSASumMag / normFactor;

            if (solverPerf.checkConvergence(tolerance_, relTol_))
            {
                PBiCGStab_misc_time += clock.timeIncrement();
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for (label cell=0; cell<nCells; cell++)
                {
                    psiPtr[cell] += alpha * yAPtr[cell];
                }
                solverPerf.nIterations()++;

                PBiCGStab_axpy_time += clock.timeIncrement();
                break;
            }

            // --- Precondition sA
            preconPtr_->precondition(zA, sA, cmpt);
            PBiCGStab_precondition_time += clock.timeIncrement();

            // --- Calculate tA
            matrix_.Amul(tA, zA, interfaceBouCoeffs_, interfaces_, cmpt);
            PBiCGStab_spmv_time += clock.timeIncrement();

            scalar tAtA = 0.;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:tAtA)
#endif
            for(label c = 0; c < nCells; ++c){
                tAtA += tAPtr[c] * tAPtr[c];
            }

            PBiCGStab_reduce_local_time += clock.timeIncrement();

            reduce(tAtA, sumOp<scalar>());
            
            PBiCGStab_allreduce_time += clock.timeIncrement();

            // --- Calculate omega from tA and sA
            //     (cheaper than using zA with preconditioned tA)
            // omega = gSumProd(tA, sA, matrix().mesh().comm()) / tAtA;

            omega = 0.;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:omega)
#endif
            for(label c = 0; c < nCells; ++c){
                omega += tAPtr[c] * sAPtr[c];
            }

            PBiCGStab_reduce_local_time += clock.timeIncrement();

            reduce(omega, sumOp<scalar>());

            PBiCGStab_allreduce_time += clock.timeIncrement();

            omega /= tAtA;

            // --- Update solution and residual
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (label cell=0; cell<nCells; cell++)
            {
                psiPtr[cell] += alpha * yAPtr[cell] + omega * zAPtr[cell];
                rAPtr[cell] = sAPtr[cell] - omega * tAPtr[cell];
            }

            PBiCGStab_axpy_time += clock.timeIncrement();

            gRASumMag = 0.;
            #ifdef _OPENMP
            #pragma omp parallel for reduction(+:gRASumMag)
            #endif
                for(label c = 0; c < nCells; ++c){
                    gRASumMag += std::abs(rAPtr[c]);
                }

            PBiCGStab_reduce_local_time += clock.timeIncrement();

            reduce(gRASumMag, sumOp<scalar>());

            PBiCGStab_allreduce_time += clock.timeIncrement();
            
            solverPerf.finalResidual() = gRASumMag / normFactor;

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
