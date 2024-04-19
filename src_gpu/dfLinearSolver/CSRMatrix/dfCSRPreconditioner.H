#pragma once
#include "dfMatrixDataBase.H"

class CSRPreconditioner{

public:

    CSRPreconditioner(){}
    virtual ~CSRPreconditioner() {}

    virtual void initialize() = 0;

    virtual void precondition
    (
        double *wA,
        const double *rA
    ) = 0;
};

class GAMGCSRPreconditioner : public CSRPreconditioner {
public:
    // variables related to the GAMG preconditioner
    int nLevels;

    GAMGCSRPreconditioner() {}
    virtual ~GAMGCSRPreconditioner() {}

    virtual void initialize(
        /* init variables related to GAMG preconditioner */
    ) override;

    virtual void precondition
    (
        double *wA,
        const double *rA
    ) override;
};