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

#include "CSRPBiCGStab.H"
#include <mpi.h>

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(CSRPBiCGStab, 0);

    csrMatrix::solver::addsymMatrixConstructorToTable<CSRPBiCGStab>
        addCSRPBiCGStabSymMatrixConstructorToTable_;

    csrMatrix::solver::addasymMatrixConstructorToTable<CSRPBiCGStab>
        addCSRPBiCGStabAsymMatrixConstructorToTable_;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::CSRPBiCGStab::CSRPBiCGStab
(
    const word& fieldName,
    const csrMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
:
    csrMatrix::solver
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces,
        solverControls
    ),
    preconPtr_(
        csrMatrix::preconditioner::New
        (
            *this,
            controlDict_
        )
    )
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::solverPerformance Foam::CSRPBiCGStab::solve
(
    scalarField& psi,
    const scalarField& source,
    const direction cmpt
) const
{

    double spmv_start, spmv_end;
    double normFactor_start, normFactor_end;
    double gSumMag_start, gSumMag_end;
    double gSumProd_start, gSumProd_end;
    double gSumSqr_start, gSumSqr_end;
    double localUpdate_start, localUpdate_end;
    double precondition_start, precondition_end;

    Info << "Foam::CSRPBiCGStab::solve start -------------------------------------" << endl;
    double solve_start = MPI_Wtime();

    // --- Setup class containing solver performance data
    solverPerformance solverPerf
    (
        csrMatrix::preconditioner::getName(controlDict_) + typeName,
        fieldName_
    );

    const label nCells = psi.size();

    scalar* __restrict__ psiPtr = psi.begin();

    scalarField pA(nCells);
    scalar* __restrict__ pAPtr = pA.begin();

    scalarField yA(nCells);
    scalar* __restrict__ yAPtr = yA.begin();

    // --- Calculate A.psi
    spmv_start = MPI_Wtime();
    matrix_.Amul(yA, psi, interfaceBouCoeffs_, interfaces_, cmpt);
    spmv_end = MPI_Wtime();
    spmv_time += spmv_end - spmv_start;

    // --- Calculate initial residual field
    localUpdate_start = MPI_Wtime();
    scalarField rA(source - yA);
    localUpdate_end = MPI_Wtime();
    localUpdate_time += localUpdate_end - localUpdate_start;

    scalar* __restrict__ rAPtr = rA.begin();

    // --- Calculate normalisation factor
    normFactor_start = MPI_Wtime();
    const scalar normFactor = this->normFactor(psi, source, yA, pA);
    printf("normFactor-C = %.10e\n",normFactor);
    normFactor_end = MPI_Wtime();
    normFactor_time += normFactor_end - normFactor_start;

    // --- Calculate normalised residual norm
    gSumMag_start = MPI_Wtime();
    solverPerf.initialResidual() =
        gSumMag(rA, matrix().mesh().comm())
       /normFactor;
    gSumMag_end = MPI_Wtime();
    gSumMag_time += gSumMag_end - gSumMag_start;

    solverPerf.finalResidual() = solverPerf.initialResidual();

    printf("finalResidual()-C = %.10e\n",solverPerf.finalResidual());

    // --- Check convergence, solve if not converged
    if
    (
        minIter_ > 0
     || !solverPerf.checkConvergence(tolerance_, relTol_)
    )
    {
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

            gSumProd_start = MPI_Wtime();
            rA0rA = gSumProd(rA0, rA, matrix().mesh().comm());
            gSumProd_end = MPI_Wtime();
            gSumProd_time += gSumProd_end - gSumProd_start;

            printf("--- rA0rA = %.5e and nterations = %d\n",rA0rA,solverPerf.nIterations());


            // --- Test for singularity
            if (solverPerf.checkSingularity(mag(rA0rA)))
            {
                break;
            }

            // --- Update pA
            if (solverPerf.nIterations() == 0)
            {
                localUpdate_start = MPI_Wtime();
                for (label cell=0; cell<nCells; cell++)
                {
                    pAPtr[cell] = rAPtr[cell];
                }
                localUpdate_end = MPI_Wtime();
                localUpdate_time += localUpdate_end - localUpdate_start;
            }
            else
            {
                // --- Test for singularity
                if (solverPerf.checkSingularity(mag(omega)))
                {
                    break;
                }

                localUpdate_start = MPI_Wtime();
                const scalar beta = (rA0rA/rA0rAold)*(alpha/omega);
                for (label cell=0; cell<nCells; cell++)
                {
                    pAPtr[cell] =
                        rAPtr[cell] + beta*(pAPtr[cell] - omega*AyAPtr[cell]);
                }
                localUpdate_end = MPI_Wtime();
                localUpdate_time += localUpdate_end - localUpdate_start;
            }

            // --- Precondition pA
            precondition_start = MPI_Wtime();
            preconPtr_->precondition(yA, pA, cmpt);
            precondition_end = MPI_Wtime();
            precondition_time += precondition_end - precondition_start;
            
            // --- Calculate AyA
            spmv_start = MPI_Wtime();   
            matrix_.Amul(AyA, yA, interfaceBouCoeffs_, interfaces_, cmpt);
            spmv_end = MPI_Wtime();
            spmv_time += spmv_end - spmv_start;

            gSumProd_start = MPI_Wtime();
            const scalar rA0AyA = gSumProd(rA0, AyA, matrix().mesh().comm());
            gSumProd_end = MPI_Wtime();
            gSumProd_time += gSumProd_end - gSumProd_start;

            // --- Calculate sA
            localUpdate_start = MPI_Wtime();
            alpha = rA0rA/rA0AyA;
            printf("alpha = rA0rA/rA0AyA : %.10e, %.10e, %.10e\n", alpha, rA0rA, rA0AyA);
            // --- Calculate sA
            for (label cell=0; cell<nCells; cell++)
            {
                sAPtr[cell] = rAPtr[cell] - alpha*AyAPtr[cell];
            }
            localUpdate_end = MPI_Wtime();
            localUpdate_time += localUpdate_end - localUpdate_start;

            // --- Test sA for convergence
            gSumMag_start = MPI_Wtime();
            solverPerf.finalResidual() =
                gSumMag(sA, matrix().mesh().comm())/normFactor;
            gSumMag_end = MPI_Wtime();
            gSumMag_time += gSumMag_end - gSumMag_start;
            printf("mid finalResidual = finalResidual / normFactor : %.10e, %.10e\n", solverPerf.finalResidual(), normFactor);

            if (solverPerf.checkConvergence(tolerance_, relTol_))
            {
                localUpdate_start = MPI_Wtime();
                for (label cell=0; cell<nCells; cell++)
                {
                    psiPtr[cell] += alpha*yAPtr[cell];
                }
                solverPerf.nIterations()++;
                localUpdate_end = MPI_Wtime();
                localUpdate_time += localUpdate_end - localUpdate_start;

                break;
            }

            // --- Precondition sA
            precondition_start = MPI_Wtime();
            preconPtr_->precondition(zA, sA, cmpt);
            precondition_end = MPI_Wtime();
            precondition_time += precondition_end - precondition_start;

            // --- Calculate tA
            spmv_start = MPI_Wtime();   
            matrix_.Amul(tA, zA, interfaceBouCoeffs_, interfaces_, cmpt);
            spmv_end = MPI_Wtime();
            spmv_time += spmv_end - spmv_start;

            gSumSqr_start = MPI_Wtime();
            const scalar tAtA = gSumSqr(tA, matrix().mesh().comm());
            gSumSqr_end = MPI_Wtime();
            gSumSqr_time += gSumSqr_end - gSumSqr_start;

            // --- Calculate omega from tA and sA
            //     (cheaper than using zA with preconditioned tA)
            gSumProd_start = MPI_Wtime();
            omega = gSumProd(tA, sA, matrix().mesh().comm())/tAtA;
            printf("omega = omega / tAtA : %.10e, %.10e\n", omega, tAtA);
            gSumProd_end = MPI_Wtime();
            gSumProd_time += gSumProd_end - gSumProd_start;

            // --- Update solution and residual
            localUpdate_start = MPI_Wtime();
            for (label cell=0; cell<nCells; cell++)
            {
                psiPtr[cell] += alpha*yAPtr[cell] + omega*zAPtr[cell];
                rAPtr[cell] = sAPtr[cell] - omega*tAPtr[cell];
            }
            localUpdate_end = MPI_Wtime();
            localUpdate_time += localUpdate_end - localUpdate_start;

            gSumMag_start = MPI_Wtime();
            solverPerf.finalResidual() =
                gSumMag(rA, matrix().mesh().comm())
               /normFactor;
            gSumMag_end = MPI_Wtime();
            gSumMag_time += gSumMag_end - gSumMag_start;

            printf("finalResidual = finalResidual / normFactor : %.10e,%.10e\n",solverPerf.finalResidual(), normFactor);

            printf("Initial residual = %.10f, Final residual = %.10f\n",solverPerf.initialResidual(),solverPerf.finalResidual());

            printf("tolerence = %.10e\n", tolerance_);
            printf("RelTolerance = %.10e\n", relTol_);
            printf("small*pTraits<Type>::one = %.10e\n",solverPerformance::small_ * Foam::pTraits<scalar>::one);
            printf("cmptMultiply(RelTolerance, initialResidual_) = %.10e\n",Foam::cmptMultiply(relTol_, solverPerf.initialResidual()));


        } while
        (
            (
              ++solverPerf.nIterations() < maxIter_
            && !solverPerf.checkConvergence(tolerance_, relTol_)
            )
         || solverPerf.nIterations() < minIter_
        );
    }

    double solve_end = MPI_Wtime();
    solve_time += solve_end - solve_start;

    Info << "spmv time : " << spmv_time << ", " << spmv_time / solve_time * 100 << "%" << endl;
    Info << "normFactor time : " << normFactor_time << ", " << normFactor_time / solve_time * 100 << "%" << endl;
    Info << "gSumMag time : " << gSumMag_time << ", " << gSumMag_time / solve_time * 100 << "%" << endl;
    Info << "gSumProd time : " << gSumProd_time << ", " << gSumProd_time / solve_time * 100 << "%" << endl;
    Info << "gSumSqr time : " << gSumSqr_time << ", " << gSumSqr_time / solve_time * 100 << "%" << endl;
    Info << "localUpdate time : " << localUpdate_time << ", " << localUpdate_time / solve_time * 100 << "%" << endl;
    Info << "precondition time : " << precondition_time << ", " << precondition_time / solve_time * 100 << "%" << endl;
    Info << "Foam::CSRPBiCGStab::solve end --------------------------------------------" << endl;
    return solverPerf;
}


// ************************************************************************* //
