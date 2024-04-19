#pragma once

#include "csrMatrix.H"

namespace Foam
{

class CSRPCG: public  csrMatrix::solver{

    autoPtr<csrMatrix::preconditioner> preconPtr_;


public:
    //- Runtime type information
    TypeName("PCG");

    //- Construct from matrix components and solver controls
    CSRPCG
    (
        const word& fieldName,
        const csrMatrix& matrix,
        const FieldField<Field, scalar>& interfaceBouCoeffs,
        const FieldField<Field, scalar>& interfaceIntCoeffs,
        const lduInterfaceFieldPtrsList& interfaces,
        const dictionary& solverControls
    );

    //- Disallow default bitwise copy construction
    CSRPCG(const CSRPCG&) = delete;

    //- Destructor
    virtual ~CSRPCG(){}

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
    void operator=(const CSRPCG&) = delete;


    // profiling
    mutable double solve_time = 0.;
    mutable double spmv_time = 0.;
    mutable double normFactor_time = 0.;
    mutable double gSumMag_time = 0.;
    mutable double gSumProd_time = 0.;
    mutable double localUpdate_time = 0.;
    mutable double precondition_time = 0.;
    mutable double precondition_construct_time = 0.;

    void print_time() const {
        double other_time = solve_time - spmv_time - normFactor_time - gSumMag_time - gSumProd_time - localUpdate_time - precondition_time;
        Info << "CSRPCG solve profiling -----------------------------------------------------------------" << endl;
        Info << "solve time : " << solve_time << endl;
        Info << "spmv time : " << spmv_time << ", " << spmv_time / solve_time * 100 << "%" << endl;
        Info << "normFactor time : " << normFactor_time << ", " << normFactor_time / solve_time * 100 << "%" << endl;
        Info << "gSumMag time : " << gSumMag_time << ", " << gSumMag_time / solve_time * 100 << "%" << endl;
        Info << "gSumProd time : " << gSumProd_time << ", " << gSumProd_time / solve_time * 100 << "%" << endl;
        Info << "localUpdate time : " << localUpdate_time << ", " << localUpdate_time / solve_time * 100 << "%" << endl;
        Info << "precondition time : " << precondition_time << ", " << precondition_time / solve_time * 100 << "%" << endl;
        Info << "other time : " << other_time << ", " << other_time / solve_time * 100 << "%" << endl;
        Info << "----------------------------------------------------------------------------------------" << endl;
        preconPtr_->print_time();
    }
};

}

