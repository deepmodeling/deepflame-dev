#pragma once
#include "dfMatrixDataBase.H"
#include "dfCSRPreconditioner.H"

class CSRSolver {
protected:
    double small_;
    double vsmall_;
    int maxIter_;
    int minIter_;
    double tolerance_;
    double relTol_;
public:

    CSRSolver() : small_(1e-20), vsmall_(2.22507e-308), maxIter_(100), minIter_(0), tolerance_(1e-9), relTol_(0) {}
    virtual ~CSRSolver() {}

    virtual void initialize(const int, const size_t) = 0;

    virtual bool checkSingularity(double input){
        return (input < vsmall_);
    };

    virtual bool checkConvergence
    (
        double finalResidual, 
        double initialResidual,
        int nIterations
    )
    {
        if(
            finalResidual < tolerance_ 
            || (relTol_ > small_ * 1 && finalResidual < relTol_ * initialResidual)
        )
        {
            printf("GPU-CSR-PBiCGStab::solve end --------------------------------------------\n");
            printf("Initial residual = %.5e, Final residual = %.5e, No Iterations %d\n",initialResidual,finalResidual,nIterations);
            return true;
        }
        else{
            return false;
        }
    };

    virtual void solve
    (
        const dfMatrixDataBase& dataBase,
        const double* d_internal_coeffs,
        const double* d_boundary_coeffs,
        int* patch_type,
        double* diagPtr,
        const double* off_diag_value,
        const double *rhs, 
        double *psi
    ) = 0;

    virtual void initSolvePerformance
    (
        double small, 
        double vsmall, 
        int maxIter, 
        int minIter, 
        double tolerance, 
        double relTol
    ){
        small_ = small;
        vsmall_ = vsmall;
        maxIter_ = maxIter;
        minIter_ = minIter;
        tolerance_ = tolerance;
        relTol_ = relTol;
    }
};

class PBiCGStabCSRSolver : public CSRSolver {
public:
    double *d_yA;
    double *d_rA;
    double *d_pA;
    double *d_normFactors_tmp;
    double *d_AyA;
    double *d_sA;
    double *d_zA;
    double *d_tA;
    double *d_rA0;
    double *d_rA0rA_tmp;
    double *d_rA0AyA_tmp;
    double *d_tAtA_tmp;
    double *d_sAtA_tmp;
    double *reduce_result;
    double *scalarSendBufList_;
    double *scalarRecvBufList_;

    PBiCGStabCSRSolver() {}
    virtual ~PBiCGStabCSRSolver() {}

    virtual void initialize(const int nCells, const size_t boundary_surface_value_bytes) override;

    virtual void solve
    (
        const dfMatrixDataBase& dataBase,
        const double* d_internal_coeffs,
        const double* d_boundary_coeffs,
        int* patch_type,
        double* diagPtr,
        const double* off_diag_value,
        const double *rhs, 
        double *psi
    ) override;
};

class PCGCSRSolver : public CSRSolver {
public:
    // variables related to PCG solver
    double *d_wA;
    double *d_rA;
    
    double *reduce_result;
    double *scalarSendBufList_;
    double *scalarRecvBufList_;

    // tmp
    GAMGCSRPreconditioner *precond_;


    PCGCSRSolver() {}
    virtual ~PCGCSRSolver() {}

    virtual void initialize(const int nCells, const size_t boundary_surface_value_bytes) override;

    virtual void solve
    (
        const dfMatrixDataBase& dataBase,
        const double* d_internal_coeffs,
        const double* d_boundary_coeffs,
        int* patch_type,
        double* diagPtr,
        const double* off_diag_value,
        const double *rhs, 
        double *psi
    ) override;
};