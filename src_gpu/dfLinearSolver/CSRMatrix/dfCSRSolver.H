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

    CSRSolver() : small_(1e-6), vsmall_(1e-12), maxIter_(1000), minIter_(10), tolerance_(1e-8), relTol_(1e-6) {}
    virtual ~CSRSolver() {}

    virtual void solve
    (
        const dfMatrixDataBase& dataBase,
        const double* diagPtr,
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
    PBiCGStabCSRSolver() {}
    virtual ~PBiCGStabCSRSolver() {}

    virtual void solve
    (
        const dfMatrixDataBase& dataBase,
        const double* diagPtr,
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
        const double* diagPtr,
        const double* off_diag_value,
        const double *rhs, 
        double *psi
    ) override;
};