#include "LDUPCG.H"
#include <mpi.h>

namespace Foam
{
    defineTypeNameAndDebug(LDUPCG, 0);

    LDUMatrix::solver::addsymMatrixConstructorToTable<LDUPCG>
        addLDUPCGSymMatrixConstructorToTable_;
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::LDUPCG::LDUPCG
(
    const word& fieldName,
    const LDUMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
:
    LDUMatrix::solver
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces,
        solverControls
    ),
    preconPtr_(
        LDUMatrix::preconditioner::New
        (
            *this,
            controlDict_
        )
    )
{}



// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::solverPerformance Foam::LDUPCG::solve
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
    double localUpdate_start, localUpdate_end;
    double precondition_start, precondition_end;
    double precondition_construct_start, precondition_construct_end;

    double solve_start = MPI_Wtime();

    // --- Setup class containing solver performance data
    solverPerformance solverPerf
    (
        LDUMatrix::preconditioner::getName(controlDict_) + typeName,
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

    // --- Calculate A.psi
    spmv_start = MPI_Wtime();
    matrix_.Amul(wA, psi, interfaceBouCoeffs_, interfaces_, cmpt);
    spmv_end = MPI_Wtime();
    spmv_time += spmv_end - spmv_start;

    // --- Calculate initial residual field
    localUpdate_start = MPI_Wtime();
    scalarField rA(source - wA);
    localUpdate_end = MPI_Wtime();
    localUpdate_time += localUpdate_end - localUpdate_start;

    scalar* __restrict__ rAPtr = rA.begin();

    // --- Calculate normalisation factor
    normFactor_start = MPI_Wtime();
    scalar normFactor = this->normFactor(psi, source, wA, pA);
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
            precondition_start = MPI_Wtime();
            preconPtr_->precondition(wA, rA, cmpt);
            precondition_end = MPI_Wtime();
            precondition_time += precondition_end - precondition_start;

            // --- Update search directions:
            gSumProd_start = MPI_Wtime();
            wArA = gSumProd(wA, rA, matrix().mesh().comm());
            gSumProd_end = MPI_Wtime();
            gSumProd_time += gSumProd_end - gSumProd_start;


            localUpdate_start = MPI_Wtime();
            if (solverPerf.nIterations() == 0)
            {
                for (label cell=0; cell<nCells; cell++)
                {
                    pAPtr[cell] = wAPtr[cell];
                }
            }
            else
            {
                scalar beta = wArA/wArAold;

                for (label cell=0; cell<nCells; cell++)
                {
                    pAPtr[cell] = wAPtr[cell] + beta*pAPtr[cell];
                }
            }
            localUpdate_end = MPI_Wtime();
            localUpdate_time += localUpdate_end - localUpdate_start;

            // --- Update preconditioned residual
            spmv_start = MPI_Wtime();   
            matrix_.Amul(wA, pA, interfaceBouCoeffs_, interfaces_, cmpt);
            spmv_end = MPI_Wtime();
            spmv_time += spmv_end - spmv_start;

            gSumProd_start = MPI_Wtime();
            scalar wApA = gSumProd(wA, pA, matrix().mesh().comm());
            gSumProd_end = MPI_Wtime();
            gSumProd_time += gSumProd_end - gSumProd_start;

            // --- Test for singularity
            if (solverPerf.checkSingularity(mag(wApA)/normFactor)) break;
            // --- Update solution and residual:

            localUpdate_start = MPI_Wtime();
            scalar alpha = wArA/wApA;
            for (label cell=0; cell<nCells; cell++)
            {
                psiPtr[cell] += alpha*pAPtr[cell];
                rAPtr[cell] -= alpha*wAPtr[cell];
            }
            localUpdate_end = MPI_Wtime();
            localUpdate_time += localUpdate_end - localUpdate_start;

            gSumMag_start = MPI_Wtime();
            solverPerf.finalResidual() =
                gSumMag(rA, matrix().mesh().comm())
               /normFactor;
            gSumMag_end = MPI_Wtime();
            gSumMag_time += gSumMag_end - gSumMag_start;

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

    return solverPerf;
}


// ************************************************************************* //
