#pragma once
#include "dfMatrixDataBase.H"

struct GAMGStruct
{
    int curLevel = -100; // current level && flag to save data 
    int nCell;
    int nFace;

    // === host data get from agglomeration
    std::vector<int> upperAddr;
    std::vector<int> lowerAddr;

    std::vector<int> restrictMap;
    std::vector<int> faceRestrictMap;
    std::vector<int> faceFlipMap;   // bool2int

    // === d2h matrix data for verify
    double* h_lower = nullptr;
    double* h_upper = nullptr;
    double* h_diag = nullptr;

    // === device map data
    int* d_restrictMap = nullptr;
    int* d_faceRestrictMap = nullptr;
    int* d_faceFlipMap = nullptr;   // bool2int

    // === device matrix data
    double* d_lower = nullptr;
    double* d_upper = nullptr;
    double* d_diag = nullptr;
    double* d_internal_coeffs = nullptr;
    double* d_boundary_coeffs = nullptr;

    // === device iteration data
    double* d_CorrFields = nullptr;
    double* d_Sources = nullptr;

    // === device temp data
    double* d_scalingFactorNum = nullptr;
    double* d_scalingFactorDenom = nullptr;
};

class CSRPreconditioner{

public:

    CSRPreconditioner(){}
    virtual ~CSRPreconditioner() {}

    virtual void initialize(GAMGStruct *GAMGdata_, int agglomeration_level) = 0;

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
        GAMGStruct *GAMGdata_, int agglomeration_level
    ) override;

    virtual void precondition
    (
        double *wA,
        const double *rA
    ) override;

    virtual void agglomerateMatrix(const dfMatrixDataBase& dataBase, GAMGStruct *GAMGdata_, int agglomeration_level);

    virtual void fine2coarse(const dfMatrixDataBase& dataBase, GAMGStruct *GAMGdata_, int agglomeration_level, int startLevel, int endLevel);
    virtual void coarse2fine(const dfMatrixDataBase& dataBase, GAMGStruct *GAMGdata_, int agglomeration_level, int startLevel, int endLevel);

    virtual void Vcycle(const dfMatrixDataBase& dataBase, GAMGStruct *GAMGdata_, int agglomeration_level);
};

