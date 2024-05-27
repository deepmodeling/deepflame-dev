#pragma once
#include "dfMatrixDataBase.H"
#include "dfCSRSmoother.H"

class CSRPreconditioner{

public:

    CSRPreconditioner(){}
    virtual ~CSRPreconditioner() {}

    virtual void initialize(const dfMatrixDataBase &dataBase, GAMGStruct *GAMGdata_, int agglomeration_level) = 0;

    virtual void precondition
    (
        double *wA,
        const double *rA,
        const dfMatrixDataBase& dataBase,
        GAMGStruct *GAMGdata_, int agglomeration_level,
        double *scalarSendBufList_, double *scalarRecvBufList_
    ) = 0;
};

class GAMGCSRPreconditioner : public CSRPreconditioner {
public:
    // variables related to the GAMG preconditioner
    int nLevels;

    CSRSmoother *smoother = nullptr;

    GAMGCSRPreconditioner() {}
    virtual ~GAMGCSRPreconditioner() {}

    virtual void initialize(
        /* init variables related to GAMG preconditioner */
        const dfMatrixDataBase &dataBase, GAMGStruct *GAMGdata_, int agglomeration_level
    )override;

    virtual void precondition
    (
        double *wA,
        const double *rA,
        const dfMatrixDataBase& dataBase,
        GAMGStruct *GAMGdata_, int agglomeration_level,
        double *scalarSendBufList_, double *scalarRecvBufList_
    ) override;

    virtual void agglomerateMatrix(const dfMatrixDataBase& dataBase, GAMGStruct *GAMGdata_, int agglomeration_level);

    virtual void fine2coarse(const dfMatrixDataBase& dataBase, GAMGStruct *GAMGdata_, int agglomeration_level, int startLevel, int endLevel, double *scalarSendBufList_, double *scalarRecvBufList_);
    virtual void coarse2fine(const dfMatrixDataBase& dataBase, GAMGStruct *GAMGdata_, int agglomeration_level, int startLevel, int endLevel, double *scalarSendBufList_, double *scalarRecvBufList_);
    virtual void directSolveCoarsest(const dfMatrixDataBase& dataBase, GAMGStruct *GAMGdata_, int agglomeration_level);

    virtual void Vcycle(const dfMatrixDataBase& dataBase, GAMGStruct *GAMGdata_, int agglomeration_level, double *scalarSendBufList_, double *scalarRecvBufList_);
    virtual void Wcycle(const dfMatrixDataBase& dataBase, GAMGStruct *GAMGdata_, int agglomeration_level, double *scalarSendBufList_, double *scalarRecvBufList_);
    virtual void Fcycle(const dfMatrixDataBase& dataBase, GAMGStruct *GAMGdata_, int agglomeration_level, double *scalarSendBufList_, double *scalarRecvBufList_);

    virtual void freeInitialize(GAMGStruct *GAMGdata, int agglomeration_level);

    virtual void initCycle(GAMGStruct *GAMGdata, int agglomeration_level);
};

