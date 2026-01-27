#include "dfPCG.H"
#include <mpi.h>
#include "dfArrayOp.H"
#include "PstreamReduceOps.H"

namespace Foam
{
    defineTypeNameAndDebug(dfPCG, 0);

    dfMatrix::solver::addsymMatrixConstructorToTable<dfPCG>
        adddfPCGSymMatrixConstructorToTable_;
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::dfPCG::dfPCG
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

Foam::solverPerformance Foam::dfPCG::solve
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

    label nCells = psi.size();

    scalar* __restrict__ psiPtr = psi.begin();

    scalarField pA(nCells);
    scalar* __restrict__ pAPtr = pA.begin();

    scalarField wA(nCells);
    scalar* __restrict__ wAPtr = wA.begin();

    scalarField rA(nCells);
    scalar* __restrict__ rAPtr = rA.begin();

    const scalar* __restrict__ const sourcePtr = source.begin();

    scalar wArA = solverPerf.great_;
    scalar wArAold = wArA;

    misc_time += solveClock.timeIncrement();

    // --- Calculate A.psi
    matrix_.Amul(wA, psi, interfaceBouCoeffs_, interfaces_, cmpt);
    spmv_time += solveClock.timeIncrement();

    // --- Calculate initial residual field
    // scalarField rA(source - wA);
    #pragma omp parallel for
    for (label cell=0; cell<nCells; cell++)
    {
        rAPtr[cell] = sourcePtr[cell] - wAPtr[cell];
    }

    localUpdate_time += solveClock.timeIncrement();

    // --- Calculate normalisation factor
    scalar normFactor = this->normFactor(psi, source, wA, pA);
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
        // --- Solver iteration
        do
        {
            // --- Store previous wArA
            wArAold = wArA;

            // --- Precondition residual
            preconPtr_->precondition(wA, rA, cmpt);
            precondition_time += solveClock.timeIncrement();

            // --- Update search directions:
            // wArA = gSumProd(wA, rA, matrix().mesh().comm());
            wArA = dfSumProd(wA.begin(), rA.begin(), wA.size());

            gSumProd_time += solveClock.timeIncrement();

            reduce(wArA, sumOp<scalar>());

            allreduce_time += solveClock.timeIncrement();

            if (solverPerf.nIterations() == 0)
            {
                #pragma omp parallel for
                for (label cell=0; cell<nCells; cell++)
                {
                    pAPtr[cell] = wAPtr[cell];
                }
            }
            else
            {
                scalar beta = wArA/wArAold;

                #pragma omp parallel for
                for (label cell=0; cell<nCells; cell++)
                {
                    pAPtr[cell] = wAPtr[cell] + beta*pAPtr[cell];
                }
            }
            localUpdate_time += solveClock.timeIncrement();

            // --- Update preconditioned residual
            matrix_.Amul(wA, pA, interfaceBouCoeffs_, interfaces_, cmpt);
            spmv_time += solveClock.timeIncrement();

            scalar wApA = gSumProd(wA, pA, matrix().mesh().comm());
            gSumProd_time += solveClock.timeIncrement();

            // --- Test for singularity
            if (solverPerf.checkSingularity(mag(wApA)/normFactor)) break;
            // --- Update solution and residual:

            scalar alpha = wArA/wApA;
            #pragma omp parallel for
            for (label cell=0; cell<nCells; cell++)
            {
                psiPtr[cell] += alpha*pAPtr[cell];
                rAPtr[cell] -= alpha*wAPtr[cell];
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
