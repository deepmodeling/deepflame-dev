#pragma once
#include "dfMatrixDataBase.H"
#include "dfMatrixOpBase.H"

class ELLSolver {
protected:
    double small_;
    double vsmall_;
    int maxIter_;
    int minIter_;
    double tolerance_;
    double relTol_;
public:

    double* d_yA;
    double* d_rA;
    double* d_pA;
    double* d_normFactors_tmp;
    double* d_AyA;
    double* d_sA;
    double* d_zA;
    double* d_tA;
    double* d_rA0;
    double* d_rA0rA_tmp;
    double* d_rA0AyA_tmp;
    double* d_tAtA_tmp;
    double* d_sAtA_tmp;
    double *reduce_result;
    double* scalarSendBufList_;
    double* scalarRecvBufList_;


    ELLSolver() : small_(1e-20), vsmall_(2.22507e-308), maxIter_(100), minIter_(0), tolerance_(1e-9), relTol_(0.01) {}
    virtual ~ELLSolver() {}

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
    PBiCGStabELLSolver() {}
    virtual ~PBiCGStabELLSolver() {}

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
    PCGELLSolver() {}
    virtual ~PCGELLSolver() {}

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