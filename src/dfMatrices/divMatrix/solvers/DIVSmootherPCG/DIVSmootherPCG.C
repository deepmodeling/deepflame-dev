#include "DIVSmootherPCG.H"

namespace Foam
{
    defineTypeNameAndDebug(DIVSmootherPCG, 0);

    divMatrix::solver::addsymMatrixConstructorToTable<DIVSmootherPCG>
        addDIVSmootherPCGSymMatrixConstructorToTable_;
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::DIVSmootherPCG::DIVSmootherPCG
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
    )
{}



// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::solverPerformance Foam::DIVSmootherPCG::solve
(
    scalarField& psi,
    const scalarField& source,
    const direction cmpt
) const
{
    // --- Setup class containing solver performance data
    // solverPerformance solverPerf
    // (
    //     divMatrix::preconditioner::getName(controlDict_) + typeName,
    //     fieldName_
    // );
    Info << "Foam::DIVSmootherPCG::solve start" << endl;
    solverPerformance solverPerf
    (
        typeName,
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
    matrix_.Amul(wA, psi, interfaceBouCoeffs_, interfaces_, cmpt);

    // --- Calculate initial residual field
    scalarField rA(source - wA);
    scalar* __restrict__ rAPtr = rA.begin();

    // --- Calculate normalisation factor
    scalar normFactor = this->normFactor(psi, source, wA, pA);

    // --- Calculate normalised residual norm
    solverPerf.initialResidual() =
        gSumMag(rA, matrix().mesh().comm())
       /normFactor;
    solverPerf.finalResidual() = solverPerf.initialResidual();

    // --- Check convergence, solve if not converged
    if
    (
        minIter_ > 0
     || !solverPerf.checkConvergence(tolerance_, relTol_)
    )
    {
        // --- Select and construct the preconditioner
        // autoPtr<divMatrix::preconditioner> preconPtr =
        // divMatrix::preconditioner::New
        // (
        //     *this,
        //     controlDict_
        // );

        autoPtr<divMatrix::smoother> smootherPtr = divMatrix::smoother::New
        (
            fieldName_,
            matrix_,
            interfaceBouCoeffs_,
            interfaceIntCoeffs_,
            interfaces_,
            controlDict_
        );    

        // --- Solver iteration
        do
        {
            // --- Store previous wArA
            wArAold = wArA;

            // --- Precondition residual
            // preconPtr->precondition(wA, rA, cmpt);

            wA = Zero;
            smootherPtr->smooth
            (
                wA,
                rA,
                cmpt,
                2
            );

            // --- Update search directions:
            wArA = gSumProd(wA, rA, matrix().mesh().comm());

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


            // --- Update preconditioned residual
            matrix_.Amul(wA, pA, interfaceBouCoeffs_, interfaces_, cmpt);

            scalar wApA = gSumProd(wA, pA, matrix().mesh().comm());


            // --- Test for singularity
            if (solverPerf.checkSingularity(mag(wApA)/normFactor)) break;


            // --- Update solution and residual:

            scalar alpha = wArA/wApA;

            for (label cell=0; cell<nCells; cell++)
            {
                psiPtr[cell] += alpha*pAPtr[cell];
                rAPtr[cell] -= alpha*wAPtr[cell];
            }

            solverPerf.finalResidual() =
                gSumMag(rA, matrix().mesh().comm())
               /normFactor;

        } while
        (
            (
              ++solverPerf.nIterations() < maxIter_
            && !solverPerf.checkConvergence(tolerance_, relTol_)
            )
         || solverPerf.nIterations() < minIter_
        );
    }

    Info << "Foam::DIVSmootherPCG::solve end" << endl;
    return solverPerf;
}


// ************************************************************************* //
