#pragma once

#include "dfMatrix.H"

namespace Foam
{

class dfSmootherPCG: public  dfMatrix::solver{
public:
    //- Runtime type information
    TypeName("SmootherPCG");

    //- Construct from matrix components and solver controls
    dfSmootherPCG
    (
        const word& fieldName,
        const dfMatrix& matrix,
        const FieldField<Field, scalar>& interfaceBouCoeffs,
        const FieldField<Field, scalar>& interfaceIntCoeffs,
        const lduInterfaceFieldPtrsList& interfaces,
        const dictionary& solverControls
    );

    //- Disallow default bitwise copy construction
    dfSmootherPCG(const dfSmootherPCG&) = delete;

    //- Destructor
    virtual ~dfSmootherPCG(){}

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
    void operator=(const dfSmootherPCG&) = delete;

};

}

