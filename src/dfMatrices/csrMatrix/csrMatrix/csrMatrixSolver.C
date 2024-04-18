#include "csrMatrix.H"
#include "csrDiagonalSolver.H"
#include "CSRPBiCGStab.H"
#include "CSRPCG.H"
#include "CSRSmootherPCG.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineRunTimeSelectionTable(csrMatrix::solver, symMatrix);
    defineRunTimeSelectionTable(csrMatrix::solver, asymMatrix);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

Foam::autoPtr<Foam::csrMatrix::solver> Foam::csrMatrix::solver::New
(
    const word& fieldName,
    const csrMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
{
    const word name(solverControls.lookup("solver"));

    if (matrix.diagonal())
    {
        return autoPtr<csrMatrix::solver>
        (
            new csrDiagonalSolver
            (
                fieldName,
                matrix,
                interfaceBouCoeffs,
                interfaceIntCoeffs,
                interfaces,
                solverControls
            )
        );
    }
    else if (matrix.symmetric())
    {
        symMatrixConstructorTable::iterator constructorIter =
            symMatrixConstructorTablePtr_->find(name);

        if (constructorIter == symMatrixConstructorTablePtr_->end())
        {
            FatalIOErrorInFunction(solverControls)
                << "Unknown symmetric matrix solver " << name << nl << nl
                << "Valid symmetric matrix solvers are :" << endl
                << symMatrixConstructorTablePtr_->sortedToc()
                << exit(FatalIOError);
        }

        return autoPtr<csrMatrix::solver>
        (
            constructorIter()
            (
                fieldName,
                matrix,
                interfaceBouCoeffs,
                interfaceIntCoeffs,
                interfaces,
                solverControls
            )
        );

        // return autoPtr<csrMatrix::solver>
        // (
        //     new CSRSmootherPCG
        //     (
        //         fieldName,
        //         matrix,
        //         interfaceBouCoeffs,
        //         interfaceIntCoeffs,
        //         interfaces,
        //         solverControls
        //     )
        // );
    }
    else if (matrix.asymmetric())
    {
        asymMatrixConstructorTable::iterator constructorIter =
            asymMatrixConstructorTablePtr_->find(name);

        if (constructorIter == asymMatrixConstructorTablePtr_->end())
        {
            FatalIOErrorInFunction(solverControls)
                << "Unknown asymmetric matrix solver " << name << nl << nl
                << "Valid asymmetric matrix solvers are :" << endl
                << asymMatrixConstructorTablePtr_->sortedToc()
                << exit(FatalIOError);
        }
        return autoPtr<csrMatrix::solver>
        (
            constructorIter()
            (
                fieldName,
                matrix,
                interfaceBouCoeffs,
                interfaceIntCoeffs,
                interfaces,
                solverControls
            )
        );
        // return autoPtr<csrMatrix::solver>
        // (
        //     new CSRPBiCGStab
        //     (
        //         fieldName,
        //         matrix,
        //         interfaceBouCoeffs,
        //         interfaceIntCoeffs,
        //         interfaces,
        //         solverControls
        //     )
        // );
    }
    else
    {
        FatalIOErrorInFunction(solverControls)
            << "cannot solve incomplete matrix, "
               "no diagonal or off-diagonal coefficient"
            << exit(FatalIOError);

        return autoPtr<csrMatrix::solver>(nullptr);
    }
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::csrMatrix::solver::solver
(
    const word& fieldName,
    const csrMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
:
    fieldName_(fieldName),
    matrix_(matrix),
    interfaceBouCoeffs_(interfaceBouCoeffs),
    interfaceIntCoeffs_(interfaceIntCoeffs),
    interfaces_(interfaces),
    controlDict_(solverControls)
{
    readControls();
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::csrMatrix::solver::readControls()
{
    maxIter_ = controlDict_.lookupOrDefault<label>("maxIter", defaultMaxIter_);
    minIter_ = controlDict_.lookupOrDefault<label>("minIter", 0);
    tolerance_ = controlDict_.lookupOrDefault<scalar>("tolerance", 1e-6);
    relTol_ = controlDict_.lookupOrDefault<scalar>("relTol", 0);
}


void Foam::csrMatrix::solver::read(const dictionary& solverControls)
{
    controlDict_ = solverControls;
    readControls();
}


Foam::scalar Foam::csrMatrix::solver::normFactor
(
    const scalarField& psi,
    const scalarField& source,
    const scalarField& Apsi,
    scalarField& tmpField
) const
{
    // --- Calculate A dot reference value of psi
    matrix_.sumA(tmpField, interfaceBouCoeffs_, interfaces_);

    tmpField *= gAverage(psi, matrix_.mesh().comm());

    printf("psi_ave after reduce -C = %.10e\n",gAverage(psi, matrix_.mesh().comm()));

    return
        gSum
        (
            (mag(Apsi - tmpField) + mag(source - tmpField))(),
            matrix_.mesh().comm()
        )
      + solverPerformance::small_;

    // At convergence this simpler method is equivalent to the above
    // return 2*gSumMag(source) + solverPerformance::small_;
}


// ************************************************************************* //
