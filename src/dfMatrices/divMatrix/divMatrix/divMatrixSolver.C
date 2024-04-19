#include "divMatrix.H"
#include "divDiagonalSolver.H"
#include "DIVPBiCGStab.H"
#include "DIVPCG.H"
#include "DIVSmootherPCG.H"
#include "clockTime.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineRunTimeSelectionTable(divMatrix::solver, symMatrix);
    defineRunTimeSelectionTable(divMatrix::solver, asymMatrix);
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

Foam::autoPtr<Foam::divMatrix::solver> Foam::divMatrix::solver::New
(
    const word& fieldName,
    const divMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
{
    const word name(solverControls.lookup("solver"));

    if (matrix.diagonal())
    {
        return autoPtr<divMatrix::solver>
        (
            new divDiagonalSolver
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

        return autoPtr<divMatrix::solver>
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

        // return autoPtr<divMatrix::solver>
        // (
        //     new DIVSmootherPCG
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
        return autoPtr<divMatrix::solver>
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
        // return autoPtr<divMatrix::solver>
        // (
        //     new DIVPBiCGStab
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

        return autoPtr<divMatrix::solver>(nullptr);
    }
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::divMatrix::solver::solver
(
    const word& fieldName,
    const divMatrix& matrix,
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

void Foam::divMatrix::solver::readControls()
{
    maxIter_ = controlDict_.lookupOrDefault<label>("maxIter", defaultMaxIter_);
    minIter_ = controlDict_.lookupOrDefault<label>("minIter", 0);
    tolerance_ = controlDict_.lookupOrDefault<scalar>("tolerance", 1e-6);
    relTol_ = controlDict_.lookupOrDefault<scalar>("relTol", 0);
}


void Foam::divMatrix::solver::read(const dictionary& solverControls)
{
    controlDict_ = solverControls;
    readControls();
}


Foam::scalar Foam::divMatrix::solver::normFactor
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

    tmp<scalarField> tmp = mag(Apsi - tmpField) + mag(source - tmpField);



    return gSum(tmp(), matrix_.mesh().comm()) + solverPerformance::small_;
    // At convergence this simpler method is equivalent to the above
    // return 2*gSumMag(source) + solverPerformance::small_;
}


// ************************************************************************* //
