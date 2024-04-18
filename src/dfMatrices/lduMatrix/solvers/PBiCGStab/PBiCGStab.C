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

#include "PBiCGStab.H"
#include <mpi.h>

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

// namespace Foam
// {
//     defineTypeNameAndDebug(PBiCGStab, 0);

//     lduMatrix::solver::addsymMatrixConstructorToTable<PBiCGStab>
//         addPBiCGStabSymMatrixConstructorToTable_;

//     lduMatrix::solver::addasymMatrixConstructorToTable<PBiCGStab>
//         addPBiCGStabAsymMatrixConstructorToTable_;
// }


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::PBiCGStab::PBiCGStab
(
    const word& fieldName,
    const lduMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
:
    lduMatrix::solver
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces,
        solverControls
    )
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::solverPerformance Foam::PBiCGStab::solve
(
    scalarField& psi,
    const scalarField& source,
    const direction cmpt
) const
{
    double spmv_start, spmv_end, spmv_time = 0;
    int spmv_count = 0;
    double normFactor_start, normFactor_end, normFactor_time = 0;
    int normFactor_count = 0;
    double gSumMag_start, gSumMag_end, gSumMag_time = 0;
    int gSumMag_count = 0;
    double gSumProd_start, gSumProd_end, gSumProd_time = 0;
    int gSumProd_count = 0;
    double gSumSqr_start, gSumSqr_end, gSumSqr_time = 0;
    int gSumSqr_count = 0;
    double localUpdate_start, localUpdate_end, localUpdate_time = 0;
    int localUpdate_count = 0;
    double precondition_start, precondition_end, precondition_time = 0;
    int precondition_count = 0;
    double checkSingularity_start, checkSingularity_end, checkSingularity_time = 0;
    int checkSingularity_count = 0;

    // Info << "Foam::PBiCGStab::solve LDU start -------------------------------------" << endl;
    double solve_start = MPI_Wtime();

    // --- Setup class containing solver performance data
    solverPerformance solverPerf
    (
        lduMatrix::preconditioner::getName(controlDict_) + typeName,
        fieldName_
    );

    const label nCells = psi.size();

    scalar* __restrict__ psiPtr = psi.begin();

    scalarField pA(nCells);
    scalar* __restrict__ pAPtr = pA.begin();

    scalarField yA(nCells);
    scalar* __restrict__ yAPtr = yA.begin();

    // --- Calculate A.psi
    normFactor_start = MPI_Wtime();
    matrix_.Amul(yA, psi, interfaceBouCoeffs_, interfaces_, cmpt);
    normFactor_end = MPI_Wtime();
    normFactor_time += normFactor_end - normFactor_start;
    normFactor_count += 1;

    // --- Calculate initial residual field
    localUpdate_start = MPI_Wtime();
    scalarField rA(source - yA);
    localUpdate_end = MPI_Wtime();
    localUpdate_time += localUpdate_end - localUpdate_start;
    localUpdate_count += 1;

    scalar* __restrict__ rAPtr = rA.begin();

    // --- Calculate normalisation factor
    spmv_start = MPI_Wtime();
    const scalar normFactor = this->normFactor(psi, source, yA, pA);
    spmv_end = MPI_Wtime();
    spmv_time += spmv_end - spmv_start;
    spmv_count += 1;

    // --- Calculate normalised residual norm
    gSumMag_start = MPI_Wtime();
    solverPerf.initialResidual() =
        gSumMag(rA, matrix().mesh().comm())
       /normFactor;
    gSumMag_end = MPI_Wtime();
    gSumMag_time += gSumMag_end - gSumMag_start;
    gSumMag_count += 1;

    solverPerf.finalResidual() = solverPerf.initialResidual();

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

        // --- Select and construct the preconditioner
        autoPtr<lduMatrix::preconditioner> preconPtr =
        lduMatrix::preconditioner::New
        (
            *this,
            controlDict_
        );

        // --- Solver iteration
        do
        {
            // --- Store previous rA0rA
            const scalar rA0rAold = rA0rA;

            gSumProd_start = MPI_Wtime();
            rA0rA = gSumProd(rA0, rA, matrix().mesh().comm());
            gSumProd_end = MPI_Wtime();
            gSumProd_time += gSumProd_end - gSumProd_start;
            gSumProd_count += 1;


            // --- Test for singularity
            checkSingularity_start = MPI_Wtime();
            if (solverPerf.checkSingularity(mag(rA0rA)))
            {
                break;
            }
            checkSingularity_end = MPI_Wtime();
            checkSingularity_time += checkSingularity_end - checkSingularity_start;
            checkSingularity_count += 1;   

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
                localUpdate_count += 1;
            }
            else
            {
                // --- Test for singularity
                checkSingularity_start = MPI_Wtime();
                if (solverPerf.checkSingularity(mag(omega)))
                {
                    break;
                }
                checkSingularity_end = MPI_Wtime();
                checkSingularity_time += checkSingularity_end - checkSingularity_start;
                checkSingularity_count += 1;   

                localUpdate_start = MPI_Wtime();
                const scalar beta = (rA0rA/rA0rAold)*(alpha/omega);
                for (label cell=0; cell<nCells; cell++)
                {
                    pAPtr[cell] =
                        rAPtr[cell] + beta*(pAPtr[cell] - omega*AyAPtr[cell]);
                }
                localUpdate_end = MPI_Wtime();
                localUpdate_time += localUpdate_end - localUpdate_start;
                localUpdate_count += 1;
            }

            // --- Precondition pA
            precondition_start = MPI_Wtime();
            preconPtr->precondition(yA, pA, cmpt);
            precondition_end = MPI_Wtime();
            precondition_time += precondition_end - precondition_start;
            precondition_count += 1;
            
            // --- Calculate AyA
            spmv_start = MPI_Wtime();   
            matrix_.Amul(AyA, yA, interfaceBouCoeffs_, interfaces_, cmpt);
            spmv_end = MPI_Wtime();
            spmv_time += spmv_end - spmv_start;
            spmv_count += 1;

            gSumProd_start = MPI_Wtime();
            const scalar rA0AyA = gSumProd(rA0, AyA, matrix().mesh().comm());
            gSumProd_end = MPI_Wtime();
            gSumProd_time += gSumProd_end - gSumProd_start;
            gSumProd_count += 1;

            // --- Calculate sA
            localUpdate_start = MPI_Wtime();
            alpha = rA0rA/rA0AyA;
            // --- Calculate sA
            for (label cell=0; cell<nCells; cell++)
            {
                sAPtr[cell] = rAPtr[cell] - alpha*AyAPtr[cell];
            }
            localUpdate_end = MPI_Wtime();
            localUpdate_time += localUpdate_end - localUpdate_start;
            localUpdate_count += 1;

            // --- Test sA for convergence
            gSumMag_start = MPI_Wtime();
            solverPerf.finalResidual() =
                gSumMag(sA, matrix().mesh().comm())/normFactor;
            gSumMag_end = MPI_Wtime();
            gSumMag_time += gSumMag_end - gSumMag_start;
            gSumMag_count += 1;

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
                localUpdate_count += 1;

                break;
            }

            // --- Precondition sA
            precondition_start = MPI_Wtime();
            preconPtr->precondition(zA, sA, cmpt);
            precondition_end = MPI_Wtime();
            precondition_time += precondition_end - precondition_start;
            precondition_count += 1;

            // --- Calculate tA
            spmv_start = MPI_Wtime();   
            matrix_.Amul(tA, zA, interfaceBouCoeffs_, interfaces_, cmpt);
            spmv_end = MPI_Wtime();
            spmv_time += spmv_end - spmv_start;
            spmv_count += 1;

            gSumSqr_start = MPI_Wtime();
            const scalar tAtA = gSumSqr(tA, matrix().mesh().comm());
            gSumSqr_end = MPI_Wtime();
            gSumSqr_time += gSumSqr_end - gSumSqr_start;
            gSumSqr_count += 1;

            // --- Calculate omega from tA and sA
            //     (cheaper than using zA with preconditioned tA)
            gSumProd_start = MPI_Wtime();
            omega = gSumProd(tA, sA, matrix().mesh().comm())/tAtA;
            gSumProd_end = MPI_Wtime();
            gSumProd_time += gSumProd_end - gSumProd_start;
            gSumProd_count += 1;

            // --- Update solution and residual
            localUpdate_start = MPI_Wtime();
            for (label cell=0; cell<nCells; cell++)
            {
                psiPtr[cell] += alpha*yAPtr[cell] + omega*zAPtr[cell];
                rAPtr[cell] = sAPtr[cell] - omega*tAPtr[cell];
            }
            localUpdate_end = MPI_Wtime();
            localUpdate_time += localUpdate_end - localUpdate_start;
            localUpdate_count += 1;

            gSumMag_start = MPI_Wtime();
            solverPerf.finalResidual() =
                gSumMag(rA, matrix().mesh().comm())
               /normFactor;
            gSumMag_end = MPI_Wtime();
            gSumMag_time += gSumMag_end - gSumMag_start;
            gSumMag_count += 1;

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
    double solve_time = solve_end - solve_start;

    double other_time = solve_time - spmv_time - normFactor_time - gSumMag_time - gSumProd_time - gSumSqr_time - localUpdate_time - precondition_time - checkSingularity_time;

    // Info << "solve time : " << solve_time << endl;

    // Info << "spmv time : " << spmv_time << ", " << spmv_time / solve_time * 100 << "%" << endl;
    // Info << "spmv count : " << spmv_count << endl;
    // Info << "normFactor time : " << normFactor_time << ", " << normFactor_time / solve_time * 100 << "%" << endl;
    // Info << "normFactor count : " << normFactor_count << endl;
    // Info << "gSumMag time : " << gSumMag_time << ", " << gSumMag_time / solve_time * 100 << "%" << endl;
    // Info << "gSumMag count : " << gSumMag_count << endl;
    // Info << "gSumProd time : " << gSumProd_time << ", " << gSumProd_time / solve_time * 100 << "%" << endl;
    // Info << "gSumProd count : " << gSumProd_count << endl;
    // Info << "gSumSqr time : " << gSumSqr_time << ", " << gSumSqr_time / solve_time * 100 << "%" << endl;
    // Info << "gSumSqr count : " << gSumSqr_count << endl;
    // Info << "localUpdate time : " << localUpdate_time << ", " << localUpdate_time / solve_time * 100 << "%" << endl;
    // Info << "localUpdate count : " << localUpdate_count << endl;
    // Info << "precondition time : " << precondition_time << ", " << precondition_time / solve_time * 100 << "%" << endl;
    // Info << "precondition count : " << precondition_count << endl;
    // Info << "checkSingularity time : " << checkSingularity_time << ", " << checkSingularity_time / solve_time * 100 << "%" << endl;
    // Info << "checkSingularity count : " << checkSingularity_count << endl;
    // Info << "other time : " << other_time << ", " << other_time / solve_time * 100 << "%" << endl;


    // Info << "Foam::PBiCGStab::solve LDU end --------------------------------------------" << endl;
    return solverPerf;
}


// ************************************************************************* //
