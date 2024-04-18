#pragma once

#include "divMatrix.H"

namespace Foam
{

class DIVPCG: public  divMatrix::solver{

    autoPtr<divMatrix::preconditioner> preconPtr_;


public:
    //- Runtime type information
    TypeName("PCG");

    //- Construct from matrix components and solver controls
    DIVPCG
    (
        const word& fieldName,
        const divMatrix& matrix,
        const FieldField<Field, scalar>& interfaceBouCoeffs,
        const FieldField<Field, scalar>& interfaceIntCoeffs,
        const lduInterfaceFieldPtrsList& interfaces,
        const dictionary& solverControls
    );

    //- Disallow default bitwise copy construction
    DIVPCG(const DIVPCG&) = delete;

    //- Destructor
    virtual ~DIVPCG(){}

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
    void operator=(const DIVPCG&) = delete;


    // profiling
    mutable double PCG_time = 0.;
    mutable double PCG_spmv_time = 0.;
    mutable double PCG_sumA_time = 0.;
    mutable double PCG_axpy_time = 0.;
    mutable double PCG_reduce_local_time = 0.;
    mutable double PCG_allreduce_time = 0.;
    mutable double PCG_norm_local_time = 0.;
    mutable double PCG_precondition_time = 0.;
    mutable double PCG_misc_time = 0.;

    void print_time() const {
        double PCG_other_time = PCG_time - PCG_spmv_time - PCG_sumA_time - PCG_axpy_time - PCG_reduce_local_time - PCG_allreduce_time - PCG_norm_local_time - PCG_precondition_time;
        Info << "DIVPCG solve profiling -----------------------------------------------------------------" << endl;
        Info << "PCG time : " << PCG_time << endl;
        Info << "PCG spmv time : " << PCG_spmv_time << ", " << PCG_spmv_time / PCG_time * 100 << "%" << endl;
        Info << "PCG sumA time : " << PCG_sumA_time << ", " << PCG_sumA_time / PCG_time * 100 << "%" << endl;
        Info << "PCG axpy time : " << PCG_axpy_time << ", " << PCG_axpy_time / PCG_time * 100 << "%" << endl;
        Info << "PCG reduce local time : " << PCG_reduce_local_time << ", " << PCG_reduce_local_time / PCG_time * 100 << "%" << endl;
        Info << "PCG allreduce time : " << PCG_allreduce_time << ", " << PCG_allreduce_time / PCG_time * 100 << "%" << endl;
        Info << "PCG norm local time : " << PCG_norm_local_time << ", " << PCG_norm_local_time / PCG_time * 100 << "%" << endl;
        Info << "PCG precondition time : " << PCG_precondition_time << ", " << PCG_precondition_time / PCG_time * 100 << "%" << endl;
        Info << "PCG misc time : " << PCG_misc_time << ", " << PCG_misc_time / PCG_time * 100 << "%" << endl;
        Info << "PCG other time : " << PCG_other_time << ", " << PCG_other_time / PCG_time * 100 << "%" << endl;
        Info << "----------------------------------------------------------------------------------------" << endl;
        preconPtr_->print_time();

    }
};

}

