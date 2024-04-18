#include "DIVPCG.H"
#include <mpi.h>

namespace Foam
{
    defineTypeNameAndDebug(DIVPCG, 0);

    divMatrix::solver::addsymMatrixConstructorToTable<DIVPCG>
        addDIVPCGSymMatrixConstructorToTable_;
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::DIVPCG::DIVPCG
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

Foam::solverPerformance Foam::DIVPCG::solve
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

    label nCells = psi.size();

    scalar* __restrict__ psiPtr = psi.begin();

    scalarField pA(nCells);
    scalar* __restrict__ pAPtr = pA.begin();

    scalarField wA(nCells);
    scalar* __restrict__ wAPtr = wA.begin();

    scalar wArA = solverPerf.great_;
    scalar wArAold = wArA;

    PCG_misc_time += clock.timeIncrement();

    // --- Calculate A.psi
    matrix_.Amul(wA, psi, interfaceBouCoeffs_, interfaces_, cmpt);

    PCG_spmv_time += clock.timeIncrement();

    // --- Calculate initial residual field
    scalarField rA(source - wA);

    PCG_localUpdate_time += clock.timeIncrement();

    scalar* __restrict__ rAPtr = rA.begin();

    // --- Calculate normalisation factor
    scalar normFactor = this->normFactor(psi, source, wA, pA);
    PCG_normFactor_time += clock.timeIncrement();

    // --- Calculate normalised residual norm
    solverPerf.initialResidual() = gSumMag(rA, matrix().mesh().comm()) / normFactor;
    PCG_gSumMag_time += clock.timeIncrement();

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
            PCG_precondition_time += clock.timeIncrement();

            // --- Update search directions:
            wArA = gSumProd(wA, rA, matrix().mesh().comm());
            PCG_gSumProd_time += clock.timeIncrement();


            if (solverPerf.nIterations() == 0)
            {
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for (label cell=0; cell<nCells; cell++)
                {
                    pAPtr[cell] = wAPtr[cell];
                }
            }
            else
            {
                scalar beta = wArA/wArAold;
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for (label cell=0; cell<nCells; cell++)
                {
                    pAPtr[cell] = wAPtr[cell] + beta * pAPtr[cell];
                }
            }
            PCG_localUpdate_time += clock.timeIncrement();

            // --- Update preconditioned residual
            matrix_.Amul(wA, pA, interfaceBouCoeffs_, interfaces_, cmpt);
            PCG_spmv_time += clock.timeIncrement();

            scalar wApA = gSumProd(wA, pA, matrix().mesh().comm());
            PCG_gSumProd_time += clock.timeIncrement();

            // --- Test for singularity
            if (solverPerf.checkSingularity(mag(wApA)/normFactor)) break;
            // --- Update solution and residual:

            scalar alpha = wArA/wApA;
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for (label cell=0; cell<nCells; cell++)
            {
                psiPtr[cell] += alpha * pAPtr[cell];
                rAPtr[cell]  -= alpha * wAPtr[cell];
            }
            PCG_localUpdate_time += clock.timeIncrement();

            solverPerf.finalResidual() = gSumMag(rA, matrix().mesh().comm()) / normFactor;
            
            PCG_gSumMag_time += clock.timeIncrement();
        } while
        (
            (
              ++solverPerf.nIterations() < maxIter_
            && !solverPerf.checkConvergence(tolerance_, relTol_)
            )
         || solverPerf.nIterations() < minIter_
        );

    }
    PCG_time += clock.elapsedTime();

    return solverPerf;
}


// ************************************************************************* //
