#pragma once
#include "dfMatrixDataBase.H"

class ELLSolver {
protected:
    double small_;
    double vsmall_;
    int maxIter_;
    int minIter_;
    double tolerance_;
    double relTol_;
public:

    ELLSolver() : small_(1e-20), vsmall_(2.22507e-308), maxIter_(100), minIter_(0), tolerance_(1e-9), relTol_(0) {}
    virtual ~ELLSolver() {}

    virtual void initialize(const int, const size_t) = 0;

    virtual void initializeStream(const int, const size_t, cudaStream_t) = 0;

    virtual bool checkSingularity(double input){
        return (input < vsmall_);
    }

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
            printf("GPU-ELL::solve end --------------------------------------------\n");
            printf("Initial residual = %.5e, Final residual = %.5e, No Iterations %d\n",initialResidual,finalResidual,nIterations);
            return true;
        }
        else{
            return false;
        }
    }

    virtual void solve
    (
        const dfMatrixDataBase& dataBase,
        const double* d_internal_coeffs,
        const double* d_boundary_coeffs,
        int* patch_type,
        double* diagPtr,
        double* ellValues,
        int* ellCols,
        int ell_max_count,
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

class PBiCGStabELLSolver : public ELLSolver {
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

    PBiCGStabELLSolver() {}
    virtual ~PBiCGStabELLSolver() {}

    virtual void initialize(const int nCells, const size_t boundary_surface_value_bytes) override;

    virtual void initializeStream(const int nCells, const size_t boundary_surface_value_bytes, cudaStream_t stream) override;

    virtual void solve
    (
        const dfMatrixDataBase& dataBase,
        const double* d_internal_coeffs,
        const double* d_boundary_coeffs,
        int* patch_type,
        double* diagPtr,
        double* ellValues,
        int* ellCols,
        int ell_max_count,
        const double *rhs, 
        double *psi
    ) override;
};

class PCGELLSolver : public ELLSolver {
public:
    // variables related to PCG solver
    double *d_wA;
    double *d_rA;
    double *d_pA;
    double *d_normFactors_tmp;
    double *d_wArA_tmp;
    double *d_wApA_tmp;
    
    double *reduce_result;
    double *scalarSendBufList_;
    double *scalarRecvBufList_;

    PCGELLSolver() {}
    virtual ~PCGELLSolver() {}

    virtual void initialize(const int nCells, const size_t boundary_surface_value_bytes) override;

    virtual void initializeStream(const int nCells, const size_t boundary_surface_value_bytes, cudaStream_t stream) override;

    virtual void solve
    (
        const dfMatrixDataBase& dataBase,
        const double* d_internal_coeffs,
        const double* d_boundary_coeffs,
        int* patch_type,
        double* diagPtr,
        double* ellValues,
        int* ellCols,
        int ell_max_count,
        const double *rhs, 
        double *psi
    ) override;
};