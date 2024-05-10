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

    // === host data for multigpu parallel
    std::vector<int> nPatchFaces;

    // === d2h matrix data for verify
    double* h_lower = nullptr;
    double* h_upper = nullptr;
    double* h_diag = nullptr;

    // === d2h interface data for verify
    double** h_interfaceBouCoeffs = nullptr;
    double** h_interfaceIntCoeffs = nullptr;

    // === device map data
    int* d_restrictMap = nullptr;
    int* d_faceRestrictMap = nullptr;
    int* d_faceFlipMap = nullptr;   // bool2int

    // === device data for multigpu parallel
    int** d_faceCells = nullptr;
    int** d_patchFaceRestrictMap = nullptr;
    double** d_interfaceBouCoeffs = nullptr;
    double** d_interfaceIntCoeffs = nullptr;
  
    // === device matrix data
    double* d_lower = nullptr;
    double* d_upper = nullptr;
    double* d_diag = nullptr;

    // === device csr matrix data
    double* d_off_diag_value = nullptr;
    int* d_csr_row_index_no_diag = nullptr;
    int* d_csr_col_index_no_diag = nullptr;

    // === device iteration data
    double* d_CorrFields = nullptr;
    double* d_Sources = nullptr;

    // === device temp data
    double* d_AcfField = nullptr;
    double* d_preSmoothField = nullptr;
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
        const double *rA,
        const dfMatrixDataBase& dataBase,
        GAMGStruct *GAMGdata_, int agglomeration_level
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
        const double *rA,
        const dfMatrixDataBase& dataBase,
        GAMGStruct *GAMGdata_, int agglomeration_level
    ) override;

    virtual void agglomerateMatrix(const dfMatrixDataBase& dataBase, GAMGStruct *GAMGdata_, int agglomeration_level);

    virtual void fine2coarse(const dfMatrixDataBase& dataBase, GAMGStruct *GAMGdata_, int agglomeration_level, int startLevel, int endLevel);
    virtual void coarse2fine(const dfMatrixDataBase& dataBase, GAMGStruct *GAMGdata_, int agglomeration_level, int startLevel, int endLevel);
    virtual void directSolveCoarsest(const dfMatrixDataBase& dataBase, GAMGStruct *GAMGdata_, int agglomeration_level);

    virtual void Vcycle(const dfMatrixDataBase& dataBase, GAMGStruct *GAMGdata_, int agglomeration_level);

    virtual void freeInitialize(GAMGStruct *GAMGdata, int agglomeration_level);
};

