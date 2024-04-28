#include "dfCSRPreconditioner.H"
#include "dfSolverOpBase.H"

// kernel functions for PBiCGStab solver

void GAMGCSRPreconditioner::initialize
(
    const dfMatrixDataBase& dataBase,
    GAMGStruct *GAMGdata, int agglomeration_level
)
{
    std::cout << "*** call in GAMGCSRPreconditioner::initialize(): init Vcycle " << std::endl;
    // for(int leveli=0; leveli<agglomeration_level; leveli++)
    // {
    //     std::cout << "malloc leveli: " << leveli << std::endl;
    //     // iteration data
    //     checkCudaErrors(cudaMalloc(&GAMGdata[leveli].d_CorrFields, GAMGdata[leveli].nCell * sizeof(double)));
    //     checkCudaErrors(cudaMalloc(&GAMGdata[leveli].d_Sources,    GAMGdata[leveli].nCell * sizeof(double)));

    //     checkCudaErrors(cudaMemset(GAMGdata[leveli].d_CorrFields, 0, GAMGdata[leveli].nCell * sizeof(double)));
    //     checkCudaErrors(cudaMemset(GAMGdata[leveli].d_Sources,    0, GAMGdata[leveli].nCell * sizeof(double)));

    //     // temp data for reduce
    //     // checkCudaErrors(cudaMalloc(&GAMGdata[leveli].d_scalingFactorNum,   GAMGdata[leveli].nCell * sizeof(double)));
    //     // checkCudaErrors(cudaMalloc(&GAMGdata[leveli].d_scalingFactorDenom, GAMGdata[leveli].nCell * sizeof(double)));
    // }
};

void GAMGCSRPreconditioner::agglomerateMatrix
(
    const dfMatrixDataBase& dataBase,
    GAMGStruct *GAMGdata_, int agglomeration_level
)
{
    std::cout << "********* call in GAMGCSRPreconditioner::agglomerateMatrix " << std::endl;
    int leveli=0;
    do
    {
        std::cout << "  level: " << leveli << ", in cell: " << GAMGdata_[leveli].nCell
                                           << ", out cell: " << GAMGdata_[leveli+1].nCell << std::endl;

        restrictFieldGPU(dataBase.stream, GAMGdata_[leveli].nCell, 
                        GAMGdata_[leveli].d_restrictMap, 
                        GAMGdata_[leveli].d_diag, GAMGdata_[leveli+1].d_diag);

        restrictMatrixGPU(dataBase.stream, GAMGdata_[leveli].nFace, 
                        GAMGdata_[leveli].d_faceRestrictMap, GAMGdata_[leveli].d_faceFlipMap,
                        GAMGdata_[leveli].d_upper, GAMGdata_[leveli].d_lower,
                        GAMGdata_[leveli+1].d_upper, GAMGdata_[leveli+1].d_lower, GAMGdata_[leveli+1].d_diag);

        //TODO: calculate interface & interfaceCoef on coarseGrid...

        leveli++; // goto coarser level

    } while( leveli < agglomeration_level-1 );
};

void GAMGCSRPreconditioner::fine2coarse
(
    const dfMatrixDataBase& dataBase,
    GAMGStruct *GAMGdata_, int agglomeration_level,
    int startLevel, int endLevel
)
{
    std::cout << "********* call in GAMGCSRPreconditioner::fine2coarse " << std::endl;
    for(int leveli=startLevel; leveli<=endLevel; leveli++)
    {

    }

};

void GAMGCSRPreconditioner::coarse2fine
(
    const dfMatrixDataBase& dataBase,
    GAMGStruct *GAMGdata_, int agglomeration_level,
    int startLevel, int endLevel
)
{
    std::cout << "********* call in GAMGCSRPreconditioner::coarse2fine " << std::endl;
    for(int leveli=startLevel; leveli>=endLevel; leveli--)
    {

    }

};

void GAMGCSRPreconditioner::Vcycle
(
    const dfMatrixDataBase& dataBase,
    GAMGStruct *GAMGdata_, int agglomeration_level
)
{
    std::cout << "********* call in GAMGCSRPreconditioner::Vcycle " << std::endl;
    fine2coarse(dataBase, GAMGdata_, agglomeration_level, 0, agglomeration_level-1);
    coarse2fine(dataBase, GAMGdata_, agglomeration_level, agglomeration_level-1, 0);
};

void GAMGCSRPreconditioner::precondition
(
    double *wA,
    const double *rA
)
{
    // Implement the GAMG precondition procedure here
    std::cout << "********* call in GAMGCSRPreconditioner::precondition " << std::endl;
};