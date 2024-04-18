#pragma once
#include "dfMatrixDataBase.H"

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

    virtual bool checkSingularity(double input) = 0;

    virtual bool checkConvergence
    (
        double finalResidual, 
        double initialResidual,
        int nIterations
    ) = 0;

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

    virtual bool checkSingularity(double input) override;

    virtual bool checkConvergence
    (
        double finalResidual, 
        double initialResidual,
        int nIterations
    ) override;

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
    PCGCSRSolver() {}
    virtual ~PCGCSRSolver() {}

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