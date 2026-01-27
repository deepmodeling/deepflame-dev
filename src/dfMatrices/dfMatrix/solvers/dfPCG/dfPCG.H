#pragma once

#include "dfMatrix.H"

namespace Foam
{

class dfPCG: public  dfMatrix::solver{

    autoPtr<dfMatrix::preconditioner> preconPtr_;


public:
    //- Runtime type information
    TypeName("PCG");

    //- Construct from matrix components and solver controls
    dfPCG
    (
        const word& fieldName,
        const dfMatrix& matrix,
        const FieldField<Field, scalar>& interfaceBouCoeffs,
        const FieldField<Field, scalar>& interfaceIntCoeffs,
        const lduInterfaceFieldPtrsList& interfaces,
        const dictionary& solverControls
    );

    //- Disallow default bitwise copy construction
    dfPCG(const dfPCG&) = delete;

    //- Destructor
    virtual ~dfPCG(){}

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
    void operator=(const dfPCG&) = delete;


    // profiling
    mutable double solve_time = 0.;
    mutable double spmv_time = 0.;
    mutable double normFactor_time = 0.;
    mutable double gSumMag_time = 0.;
    mutable double gSumProd_time = 0.;
    mutable double localUpdate_time = 0.;
    mutable double precondition_time = 0.;
    mutable double misc_time = 0;
    mutable double allreduce_time = 0.;

    void print_time() const {
        // double other_time = solve_time - spmv_time - normFactor_time - gSumMag_time - gSumProd_time - localUpdate_time - precondition_time - misc_time - allreduce_time;
        Info << "dfPCG solve profiling -----------------------------------------------------------------" << endl;
        Info << "solve time : " << solve_time << endl;
        Info << "spmv time : " << spmv_time << ", " << spmv_time / solve_time * 100 << "%" << endl;
        Info << "normFactor time : " << normFactor_time << ", " << normFactor_time / solve_time * 100 << "%" << endl;
        Info << "gSumMag time : " << gSumMag_time << ", " << gSumMag_time / solve_time * 100 << "%" << endl;
        Info << "gSumProd time : " << gSumProd_time << ", " << gSumProd_time / solve_time * 100 << "%" << endl;
        Info << "localUpdate time : " << localUpdate_time << ", " << localUpdate_time / solve_time * 100 << "%" << endl;
        Info << "precondition time : " << precondition_time << ", " << precondition_time / solve_time * 100 << "%" << endl;
        Info << "allreduce time : " << allreduce_time << ", " << allreduce_time / solve_time * 100 << "%" << endl;
        Info << "misc time : " << misc_time << ", " << misc_time / solve_time * 100 << "%" << endl;
        // Info << "other time : " << other_time << ", " << other_time / solve_time * 100 << "%" << endl;
        Info << "----------------------------------------------------------------------------------------" << endl;
        preconPtr_->print_time();
    }
};

}

