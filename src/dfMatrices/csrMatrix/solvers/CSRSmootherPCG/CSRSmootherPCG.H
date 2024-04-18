#pragma once

#include "csrMatrix.H"

namespace Foam
{

class CSRSmootherPCG: public  csrMatrix::solver{
public:
    //- Runtime type information
    TypeName("SmootherPCG");

    //- Construct from matrix components and solver controls
    CSRSmootherPCG
    (
        const word& fieldName,
        const csrMatrix& matrix,
        const FieldField<Field, scalar>& interfaceBouCoeffs,
        const FieldField<Field, scalar>& interfaceIntCoeffs,
        const lduInterfaceFieldPtrsList& interfaces,
        const dictionary& solverControls
    );

    //- Disallow default bitwise copy construction
    CSRSmootherPCG(const CSRSmootherPCG&) = delete;

    //- Destructor
    virtual ~CSRSmootherPCG(){}

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
    void operator=(const CSRSmootherPCG&) = delete;

};

}

