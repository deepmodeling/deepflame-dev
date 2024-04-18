#pragma once

#include "LDUMatrix.H"

namespace Foam
{

class LDUSmootherPCG: public  LDUMatrix::solver{
public:
    //- Runtime type information
    TypeName("SmootherPCG");

    //- Construct from matrix components and solver controls
    LDUSmootherPCG
    (
        const word& fieldName,
        const LDUMatrix& matrix,
        const FieldField<Field, scalar>& interfaceBouCoeffs,
        const FieldField<Field, scalar>& interfaceIntCoeffs,
        const lduInterfaceFieldPtrsList& interfaces,
        const dictionary& solverControls
    );

    //- Disallow default bitwise copy construction
    LDUSmootherPCG(const LDUSmootherPCG&) = delete;

    //- Destructor
    virtual ~LDUSmootherPCG(){}

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
    void operator=(const LDUSmootherPCG&) = delete;

};

}

