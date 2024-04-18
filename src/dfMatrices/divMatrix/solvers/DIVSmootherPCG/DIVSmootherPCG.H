#pragma once

#include "divMatrix.H"

namespace Foam
{

class DIVSmootherPCG: public  divMatrix::solver{
public:
    //- Runtime type information
    TypeName("SmootherPCG");

    //- Construct from matrix components and solver controls
    DIVSmootherPCG
    (
        const word& fieldName,
        const divMatrix& matrix,
        const FieldField<Field, scalar>& interfaceBouCoeffs,
        const FieldField<Field, scalar>& interfaceIntCoeffs,
        const lduInterfaceFieldPtrsList& interfaces,
        const dictionary& solverControls
    );

    //- Disallow default bitwise copy construction
    DIVSmootherPCG(const DIVSmootherPCG&) = delete;

    //- Destructor
    virtual ~DIVSmootherPCG(){}

    // Member Functions

    //- Solve the matrix with this solver
    virtual solverPerformance solve
    (
        scalarField& psi,
        const scalarField& source,
        const direction cmpt=0
    ) const;


    // Member Operators

    //- Disallow default bitwise assignment
    void operator=(const DIVSmootherPCG&) = delete;

};

}

