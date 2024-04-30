#include "dfCSRPreconditioner.H"
#include "dfSolverOpBase.H"

// kernel functions for PBiCGStab solver

void GAMGCSRPreconditioner::initialize
(
    GAMGStruct *GAMGdata, int agglomeration_level
)
{
    std::cout << "*** call in GAMGCSRPreconditioner::initialize(): init Vcycle " << std::endl;
    for(int leveli=0; leveli<agglomeration_level; leveli++)
    {
        std::cout << "   malloc leveli: " << leveli << std::endl;
        // iteration data
        checkCudaErrors(cudaMalloc(&GAMGdata[leveli].d_CorrFields, GAMGdata[leveli].nCell*sizeof(double)));
        checkCudaErrors(cudaMalloc(&GAMGdata[leveli].d_Sources,    GAMGdata[leveli].nCell*sizeof(double)));

        // temp data for reduce
        checkCudaErrors(cudaMalloc(&GAMGdata[leveli].d_AcfField,   GAMGdata[leveli].nCell*sizeof(double)));
        checkCudaErrors(cudaMalloc(&GAMGdata[leveli].d_preSmoothField,   GAMGdata[leveli].nCell*sizeof(double)));
        checkCudaErrors(cudaMalloc(&GAMGdata[leveli].d_scalingFactorNum,   GAMGdata[leveli].nCell*sizeof(double)));
        checkCudaErrors(cudaMalloc(&GAMGdata[leveli].d_scalingFactorDenom, GAMGdata[leveli].nCell*sizeof(double)));
    }
    std::cout << "*** end in GAMGCSRPreconditioner::initialize(): init Vcycle " << std::endl;
    std::cout << "*********************************************************** " << std::endl;
};

void GAMGCSRPreconditioner::agglomerateMatrix
(
    const dfMatrixDataBase& dataBase,
    GAMGStruct *GAMGdata_, int agglomeration_level
)
{
    std::cout << "********* call in GAMGCSRPreconditioner::agglomerateMatrix " << std::endl;
    for(int leveli=0; leveli<agglomeration_level-1; leveli++)
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
    }
};

void GAMGCSRPreconditioner::fine2coarse
(
    const dfMatrixDataBase& dataBase,
    GAMGStruct *GAMGdata_, int agglomeration_level,
    int startLevel, int endLevel
)
{
    std::cout << "   ****** call in GAMGCSRPreconditioner::fine2coarse " << std::endl;
    for(int leveli=startLevel; leveli<endLevel; leveli++)
    {
        std::cout << "  this level: " << leveli << ", restrict source for coarser level " << std::endl;
        restrictFieldGPU(dataBase.stream, GAMGdata_[leveli].nCell, 
                        GAMGdata_[leveli].d_restrictMap, 
                        GAMGdata_[leveli].d_Sources, GAMGdata_[leveli+1].d_Sources);
        
        checkCudaErrors(cudaMemset(GAMGdata_[leveli+1].d_CorrFields, 0, GAMGdata_[leveli+1].nCell*sizeof(double)));

        //TODO: add smoother here to get d_CorrFields for leveli+1 

        if (leveli < endLevel-1)
        {
            //TODO: add scale here for d_CorrFields leveli+1

            //TODO: add Amul to get Acf


            //Purpose: GAMGdata_[leveli+1].d_Sources -= Acf
            updateSourceFieldGPU( dataBase.stream, GAMGdata_[leveli+1].nCell, 
                                  GAMGdata_[leveli+1].d_Sources, GAMGdata_[leveli+1].d_AcfField);
        }    
    }

};

void GAMGCSRPreconditioner::coarse2fine
(
    const dfMatrixDataBase& dataBase,
    GAMGStruct *GAMGdata_, int agglomeration_level,
    int startLevel, int endLevel
)
{
    std::cout << "   ****** call in GAMGCSRPreconditioner::coarse2fine " << std::endl;
    for(int leveli=startLevel; leveli>endLevel; leveli--)
    {
        //Purpose: preSmoothedCoarseCorrField = MGCorrFields[leveli-1];
        checkCudaErrors(cudaMemcpyAsync(GAMGdata_[leveli-1].d_preSmoothField, GAMGdata_[leveli-1].d_CorrFields, 
                                        GAMGdata_[leveli-1].nCell*sizeof(double), cudaMemcpyDeviceToDevice, dataBase.stream));

        std::cout << "  this level: " << leveli << ", prolong correct for finer level " << std::endl;
        prolongFieldGPU(dataBase.stream, GAMGdata_[leveli-1].nCell, 
                        GAMGdata_[leveli-1].d_restrictMap, 
                        GAMGdata_[leveli-1].d_CorrFields, GAMGdata_[leveli].d_CorrFields);

        if (leveli < startLevel - 1)
        {
            //TODO: add scale here for leveli-1
        }
        
        if (leveli > endLevel + 1)
        {
            //Purpose: MGCorrFields[leveli] += preSmoothedCoarseCorrField;
            updateCorrFieldGPU( dataBase.stream, GAMGdata_[leveli-1].nCell, 
                                GAMGdata_[leveli-1].d_CorrFields, GAMGdata_[leveli-1].d_preSmoothField);

            //TODO: add smoother here for leveli-1
        }
    }

};

void GAMGCSRPreconditioner::directSolveCoarsest
(
    const dfMatrixDataBase& dataBase,
    GAMGStruct *GAMGdata_, int agglomeration_level
)
{
    bool solveCoarsest = false;
    if (solveCoarsest)
    {
        std::cout << "   ****** call in GAMGCSRPreconditioner::directSolveCoarsest " << std::endl;
        if (GAMGdata_[agglomeration_level-1].nCell == 1)
        {
            //directSolve1x1
            directSolve1x1GPU(dataBase.stream, GAMGdata_[agglomeration_level-1].nCell, 
                                GAMGdata_[agglomeration_level-1].d_diag, 
                                GAMGdata_[agglomeration_level-1].d_CorrFields, 
                                GAMGdata_[agglomeration_level-1].d_Sources);
        }
        else if (GAMGdata_[agglomeration_level-1].nCell == 4)
        {
            //directSolve4x4
        }
        else
        {
            std::cout << "*** Unsupported dimension for aggregation amg level ..."<< std::endl;
        }
    }
};

void GAMGCSRPreconditioner::Vcycle
(
    const dfMatrixDataBase& dataBase,
    GAMGStruct *GAMGdata_, int agglomeration_level
)
{
    fine2coarse(dataBase, GAMGdata_, agglomeration_level, 0, agglomeration_level-1);

    directSolveCoarsest(dataBase, GAMGdata_, agglomeration_level);

    coarse2fine(dataBase, GAMGdata_, agglomeration_level, agglomeration_level-1, 0);
};

void GAMGCSRPreconditioner::precondition
(
    double *d_wA,
    const double *d_rA,
    double *psi,
    const dfMatrixDataBase& dataBase,
    GAMGStruct *GAMGdata_, int agglomeration_level
)
{
    std::cout << "******************************************************" << std::endl;
    std::cout << "********* call in GAMGCSRPreconditioner::precondition " << std::endl;

    //TODO: get nVcycles from control files
    int nVcycles_ = 1; 

    for (int cycle=0; cycle<nVcycles_; cycle++)
    {
        // set GAMGdata_[0].d_Sources
        checkCudaErrors(cudaMemcpyAsync(GAMGdata_[0].d_Sources, d_rA, GAMGdata_[0].nCell*sizeof(double), cudaMemcpyDeviceToDevice, dataBase.stream));
        
        // start Vcycle
        Vcycle(dataBase, GAMGdata_, agglomeration_level);

        // use GAMGdata_[0].d_CorrFields to update psi
        updateCorrFieldGPU( dataBase.stream, GAMGdata_[0].nCell, psi, GAMGdata_[0].d_CorrFields);

        //TODO: add smoother for leveli=0, nFinestSweeps_

        if (cycle < nVcycles_-1)
        {
            // Calculate finest level residual field to update d_rA
        }
    }
    std::cout << "********** end in GAMGCSRPreconditioner::precondition " << std::endl;
    std::cout << "******************************************************" << std::endl;
};