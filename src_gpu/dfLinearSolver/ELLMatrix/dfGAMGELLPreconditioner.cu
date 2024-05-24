#include "dfELLPreconditioner.H"
#include "dfSolverOpBase.H"

#define nSweeps 2

// kernel functions for PCG solver

void GAMGELLPreconditioner::initCycle
(
    GAMGStruct *GAMGdata, int agglomeration_level                                                                
)
{
    std::cout << "*** call in GAMGELLPreconditioner::initCycle " << std::endl;
    for(int leveli=0; leveli<agglomeration_level; leveli++)
    {                                 
        checkCudaErrors(cudaMemset(GAMGdata[leveli].d_CorrFields, 0, GAMGdata[leveli].nCell*sizeof(double)));
        checkCudaErrors(cudaMemset(GAMGdata[leveli].d_Sources, 0, GAMGdata[leveli].nCell*sizeof(double)));
    }
    std::cout << "*** end in GAMGELLPreconditioner::initCycle " << std::endl;
    std::cout << "*********************************************************** " << std::endl;
};

void GAMGELLPreconditioner::initialize
(
    GAMGStruct *GAMGdata, int agglomeration_level
)
{
    std::cout << "*** call in GAMGELLPreconditioner::initialize(): init Vcycle " << std::endl;

    // Jacobi Smoother
    smoother = new ELLJacobiSmoother();

    for(int leveli=0; leveli<agglomeration_level; leveli++)
    {
        std::cout << "   malloc leveli: " << leveli << std::endl;
        // matrix data                                      
        checkCudaErrors(cudaMalloc(&GAMGdata[leveli].d_lower, GAMGdata[leveli].nFace * sizeof(double)));
        checkCudaErrors(cudaMalloc(&GAMGdata[leveli].d_upper, GAMGdata[leveli].nFace * sizeof(double)));
        checkCudaErrors(cudaMalloc(&GAMGdata[leveli].d_diag,  GAMGdata[leveli].nCell * sizeof(double)));
        checkCudaErrors(cudaMalloc(&GAMGdata[leveli].d_lowerAddr, GAMGdata[leveli].nFace * sizeof(int)));
        checkCudaErrors(cudaMalloc(&GAMGdata[leveli].d_upperAddr, GAMGdata[leveli].nFace * sizeof(int)));
        checkCudaErrors(cudaMalloc(&GAMGdata[leveli].d_ell_cols, GAMGdata[leveli].nCell * GAMGdata[leveli].ell_row_maxcount * sizeof(int)));
        checkCudaErrors(cudaMalloc(&GAMGdata[leveli].d_ell_values, GAMGdata[leveli].nCell * GAMGdata[leveli].ell_row_maxcount * sizeof(double)));

        // iteration data
        checkCudaErrors(cudaMalloc(&GAMGdata[leveli].d_CorrFields, GAMGdata[leveli].nCell*sizeof(double)));
        checkCudaErrors(cudaMalloc(&GAMGdata[leveli].d_Sources,    GAMGdata[leveli].nCell*sizeof(double)));

        // temp data for reduce
        checkCudaErrors(cudaMalloc(&GAMGdata[leveli].d_AcfField,           GAMGdata[leveli].nCell*sizeof(double)));
        checkCudaErrors(cudaMalloc(&GAMGdata[leveli].d_preSmoothField,     GAMGdata[leveli].nCell*sizeof(double)));
        checkCudaErrors(cudaMalloc(&GAMGdata[leveli].d_scalingFactorNum,   GAMGdata[leveli].nCell*sizeof(double)));
        checkCudaErrors(cudaMalloc(&GAMGdata[leveli].d_scalingFactorDenom, GAMGdata[leveli].nCell*sizeof(double)));
    }
    std::cout << "*** end in GAMGELLPreconditioner::initialize(): init Vcycle " << std::endl;
    std::cout << "*********************************************************** " << std::endl;
};

void GAMGELLPreconditioner::freeInitialize
(
    GAMGStruct *GAMGdata, int agglomeration_level
)
{
    std::cout << "*** call in GAMGELLPreconditioner::initialize(): init Vcycle " << std::endl;
    for(int leveli=0; leveli<agglomeration_level; leveli++)
    {
        std::cout << "   malloc leveli: " << leveli << std::endl;
        // matrix data                                      
        checkCudaErrors(cudaFree(GAMGdata[leveli].d_lower));
        checkCudaErrors(cudaFree(GAMGdata[leveli].d_upper));
        checkCudaErrors(cudaFree(GAMGdata[leveli].d_diag));       

        // iteration data
        checkCudaErrors(cudaFree(GAMGdata[leveli].d_CorrFields));
        checkCudaErrors(cudaFree(GAMGdata[leveli].d_Sources));

        // temp data for reduce
        checkCudaErrors(cudaFree(GAMGdata[leveli].d_AcfField));
        checkCudaErrors(cudaFree(GAMGdata[leveli].d_preSmoothField));
        checkCudaErrors(cudaFree(GAMGdata[leveli].d_scalingFactorNum));
        checkCudaErrors(cudaFree(GAMGdata[leveli].d_scalingFactorDenom));
    }
    std::cout << "*** end in GAMGELLPreconditioner::initialize(): init Vcycle " << std::endl;
    std::cout << "*********************************************************** " << std::endl;
};

void GAMGELLPreconditioner::agglomerateMatrix
(
    const dfMatrixDataBase& dataBase,
    GAMGStruct *GAMGdata_, int agglomeration_level
)
{
    std::cout << "********* call in GAMGELLPreconditioner::agglomerateMatrix " << std::endl;
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

#ifdef PARALLEL_
        // agglomerateInterfaceCoefficients
        for(int patchi=0; patchi<GAMGdata_[leveli].nPatchFaces.size(); patchi++)
        {
            if (GAMGdata_[leveli].nPatchFaces[patchi] > 0)
            {
                restrictFieldGPU(dataBase.stream, GAMGdata_[leveli].nPatchFaces[patchi], 
                                GAMGdata_[leveli].d_patchFaceRestrictMap[patchi], 
                                GAMGdata_[leveli].d_interfaceBouCoeffs[patchi], 
                                GAMGdata_[leveli+1].d_interfaceBouCoeffs[patchi]);

                restrictFieldGPU(dataBase.stream, GAMGdata_[leveli].nPatchFaces[patchi], 
                                GAMGdata_[leveli].d_patchFaceRestrictMap[patchi], 
                                GAMGdata_[leveli].d_interfaceIntCoeffs[patchi], 
                                GAMGdata_[leveli+1].d_interfaceIntCoeffs[patchi]);
            }
        }
#endif
    }
};

void GAMGELLPreconditioner::fine2coarse
(
    const dfMatrixDataBase& dataBase,
    GAMGStruct *GAMGdata_, int agglomeration_level,
    int startLevel, int endLevel
)
{
    bool scaleCorrection = true;

    std::cout << "   ****** call in GAMGELLPreconditioner::fine2coarse " << std::endl;
    for(int leveli=startLevel; leveli<endLevel; leveli++)
    {
        std::cout << "  this level: " << leveli << ", restrict source for coarser level " << std::endl;

        //Purpose: get next level (leveli+1) source
        restrictFieldGPU(dataBase.stream, GAMGdata_[leveli].nCell, 
                        GAMGdata_[leveli].d_restrictMap, 
                        GAMGdata_[leveli].d_Sources, GAMGdata_[leveli+1].d_Sources);

        //Purpose: coarseCorrFields[leveli] = 0.0;
        checkCudaErrors(cudaMemset(GAMGdata_[leveli+1].d_CorrFields, 0, GAMGdata_[leveli+1].nCell*sizeof(double)));

        //Purpose: Smooth [ A * Corr = Source ] to get d_CorrFields for leveli+1
        //TODO: write nSweeps 
        smoother->smooth(dataBase.stream, nSweeps, GAMGdata_[leveli+1].nCell, GAMGdata_[leveli+1].d_CorrFields, 
                            GAMGdata_[leveli+1].d_Sources, GAMGdata_[leveli+1].ell_row_maxcount, GAMGdata_[leveli+1].d_ell_cols,
                            GAMGdata_[leveli+1].d_ell_values, GAMGdata_[leveli+1].d_diag);

        if (leveli < endLevel - 1)
        {
            //Purpose: scale d_CorrFields leveli+1, if (matrix.symmetric())
            if (scaleCorrection) 
            {
                scaleFieldGPU_ell( dataBase, GAMGdata_[leveli+1].nCell, 
                    GAMGdata_[leveli+1].d_CorrFields, GAMGdata_[leveli+1].d_Sources, GAMGdata_[leveli+1].d_AcfField, 
                    GAMGdata_[leveli+1].d_diag, GAMGdata_[leveli+1].ell_row_maxcount,
                    GAMGdata_[leveli+1].d_ell_cols, GAMGdata_[leveli+1].d_ell_values, 
                    GAMGdata_[leveli+1].d_interfaceIntCoeffs, GAMGdata_[leveli+1].d_interfaceBouCoeffs,
                    GAMGdata_[leveli+1].d_faceCells, GAMGdata_[leveli+1].nPatchFaces, 
                    GAMGdata_[leveli+1].d_scalingFactorNum, GAMGdata_[leveli+1].d_scalingFactorDenom );
            }

            //Purpose: get Acf = A * Corr & GAMGdata_[leveli+1].d_Sources -= Acf
            updateSourceFieldGPU_ell( dataBase, GAMGdata_[leveli+1].nCell, 
                                GAMGdata_[leveli+1].d_Sources, GAMGdata_[leveli+1].d_AcfField, GAMGdata_[leveli+1].d_CorrFields,
                                GAMGdata_[leveli+1].d_diag, GAMGdata_[leveli+1].ell_row_maxcount,
                                GAMGdata_[leveli+1].d_ell_cols, GAMGdata_[leveli+1].d_ell_values, 
                                GAMGdata_[leveli+1].d_interfaceIntCoeffs, GAMGdata_[leveli+1].d_interfaceBouCoeffs,
                                GAMGdata_[leveli+1].d_faceCells, GAMGdata_[leveli+1].nPatchFaces);
        }    
    }
};

void GAMGELLPreconditioner::coarse2fine
(
    const dfMatrixDataBase& dataBase,
    GAMGStruct *GAMGdata_, int agglomeration_level,
    int startLevel, int endLevel
)
{
    bool interpolateCorrection = false;
    bool scaleCorrection = true;

    std::cout << "   ****** call in GAMGELLPreconditioner::coarse2fine " << std::endl;
    for(int leveli=startLevel; leveli>endLevel; leveli--)
    {
        std::cout << "  this level: " << leveli << ", prolong correct for finer level " << std::endl;

        //Purpose: preSmoothedCoarseCorrField = MGCorrFields[leveli-1];
        checkCudaErrors(cudaMemcpyAsync(GAMGdata_[leveli-1].d_preSmoothField, GAMGdata_[leveli-1].d_CorrFields, 
                                        GAMGdata_[leveli-1].nCell*sizeof(double), cudaMemcpyDeviceToDevice, dataBase.stream));

        //Purpose: get next level (leveli-1) corr
        prolongFieldGPU(dataBase.stream, GAMGdata_[leveli-1].nCell, 
                        GAMGdata_[leveli-1].d_restrictMap, 
                        GAMGdata_[leveli-1].d_CorrFields, GAMGdata_[leveli].d_CorrFields);

        if (interpolateCorrection)
        {
            //Purpose: interpolate correctionField for next level (leveli-1)
            interpolateFieldGPU_ell(dataBase, GAMGdata_[leveli-1].nCell, GAMGdata_[leveli].nCell, 
                    GAMGdata_[leveli-1].d_CorrFields, GAMGdata_[leveli-1].d_AcfField, 
                    GAMGdata_[leveli-1].d_diag, GAMGdata_[leveli-1].ell_row_maxcount,
                    GAMGdata_[leveli-1].d_ell_cols, GAMGdata_[leveli-1].d_ell_values,  
                    GAMGdata_[leveli-1].d_interfaceIntCoeffs, GAMGdata_[leveli-1].d_interfaceBouCoeffs, 
                    GAMGdata_[leveli-1].d_faceCells, GAMGdata_[leveli-1].nPatchFaces,
                    GAMGdata_[leveli-1].d_restrictMap, GAMGdata_[leveli].d_CorrFields);
        }

        if (leveli < startLevel && scaleCorrection)
        {
            //Purpose: scale d_CorrFields leveli-1, if (matrix.symmetric())
            scaleFieldGPU_ell( dataBase, GAMGdata_[leveli-1].nCell, 
                GAMGdata_[leveli-1].d_CorrFields, GAMGdata_[leveli-1].d_Sources, GAMGdata_[leveli-1].d_AcfField, 
                GAMGdata_[leveli-1].d_diag, GAMGdata_[leveli-1].ell_row_maxcount,
                GAMGdata_[leveli-1].d_ell_cols, GAMGdata_[leveli-1].d_ell_values,
                GAMGdata_[leveli-1].d_interfaceIntCoeffs, GAMGdata_[leveli-1].d_interfaceBouCoeffs,
                GAMGdata_[leveli-1].d_faceCells, GAMGdata_[leveli-1].nPatchFaces, 
                GAMGdata_[leveli-1].d_scalingFactorNum, GAMGdata_[leveli-1].d_scalingFactorDenom );
        }
        
        if (leveli > endLevel + 1)
        {
            //Purpose: MGCorrFields[leveli] += preSmoothedCoarseCorrField;
            updateCorrFieldGPU( dataBase.stream, GAMGdata_[leveli-1].nCell, 
                                GAMGdata_[leveli-1].d_CorrFields, GAMGdata_[leveli-1].d_preSmoothField);

            //Purpose: Smooth [ A * Corr = Source ] to get d_CorrFields for leveli-1
            //TODO: write nSweeps
            smoother->smooth(dataBase.stream, nSweeps, GAMGdata_[leveli-1].nCell, GAMGdata_[leveli-1].d_CorrFields, 
                    GAMGdata_[leveli-1].d_Sources, GAMGdata_[leveli-1].ell_row_maxcount, GAMGdata_[leveli-1].d_ell_cols,
                    GAMGdata_[leveli-1].d_ell_values, GAMGdata_[leveli-1].d_diag);

        }
    }
};

void GAMGELLPreconditioner::directSolveCoarsest
(
    const dfMatrixDataBase& dataBase,
    GAMGStruct *GAMGdata_, int agglomeration_level
)
{
    bool solveCoarsest = false;
    if (solveCoarsest)
    {
        std::cout << "   ****** call in GAMGELLPreconditioner::directSolveCoarsest " << std::endl;
        if (GAMGdata_[agglomeration_level-1].nCell == 1)
        {
            //directSolve1x1
            directSolve1x1GPU(dataBase.stream, 
                                GAMGdata_[agglomeration_level-1].d_diag, 
                                GAMGdata_[agglomeration_level-1].d_CorrFields, 
                                GAMGdata_[agglomeration_level-1].d_Sources);
        }
        else if (GAMGdata_[agglomeration_level-1].nCell == 4)
        {
            //directSolve4x4
            directSolve4x4GPU(dataBase.stream, 
                        GAMGdata_[agglomeration_level-1].d_diag, 
                        GAMGdata_[agglomeration_level-1].d_upper, 
                        GAMGdata_[agglomeration_level-1].d_lower, 
                        GAMGdata_[agglomeration_level-1].d_CorrFields, 
                        GAMGdata_[agglomeration_level-1].d_Sources);
        }
        else
        {
            std::cout << "*** Unsupported dimension for aggregation amg level ..."<< std::endl;
        }
    }
};

void GAMGELLPreconditioner::Vcycle
(
    const dfMatrixDataBase& dataBase,
    GAMGStruct *GAMGdata_, int agglomeration_level
)
{
    fine2coarse(dataBase, GAMGdata_, agglomeration_level, 0, agglomeration_level-1);

    directSolveCoarsest(dataBase, GAMGdata_, agglomeration_level);

    coarse2fine(dataBase, GAMGdata_, agglomeration_level, agglomeration_level-1, 0);
};

void GAMGELLPreconditioner::Wcycle
(
    const dfMatrixDataBase& dataBase,
    GAMGStruct *GAMGdata_, int agglomeration_level
)
{
    fine2coarse(dataBase, GAMGdata_, agglomeration_level, 0, agglomeration_level-1);

    directSolveCoarsest(dataBase, GAMGdata_, agglomeration_level);

    coarse2fine(dataBase, GAMGdata_, agglomeration_level, agglomeration_level-1, agglomeration_level-2);

    fine2coarse(dataBase, GAMGdata_, agglomeration_level, agglomeration_level-2, agglomeration_level-1);

    directSolveCoarsest(dataBase, GAMGdata_, agglomeration_level);

    coarse2fine(dataBase, GAMGdata_, agglomeration_level, agglomeration_level-1, 1);

    fine2coarse(dataBase, GAMGdata_, agglomeration_level, 1, agglomeration_level-1);

    directSolveCoarsest(dataBase, GAMGdata_, agglomeration_level);

    coarse2fine(dataBase, GAMGdata_, agglomeration_level, agglomeration_level-1, agglomeration_level-2);

    fine2coarse(dataBase, GAMGdata_, agglomeration_level, agglomeration_level-2, agglomeration_level-1);

    directSolveCoarsest(dataBase, GAMGdata_, agglomeration_level);

    coarse2fine(dataBase, GAMGdata_, agglomeration_level, agglomeration_level-1, 0);
};

void GAMGELLPreconditioner::Fcycle
(
    const dfMatrixDataBase& dataBase,
    GAMGStruct *GAMGdata_, int agglomeration_level
)
{
    fine2coarse(dataBase, GAMGdata_, agglomeration_level, 0, agglomeration_level-1);

    directSolveCoarsest(dataBase, GAMGdata_, agglomeration_level);

    coarse2fine(dataBase, GAMGdata_, agglomeration_level, agglomeration_level-1, agglomeration_level-2);

    fine2coarse(dataBase, GAMGdata_, agglomeration_level, agglomeration_level-2, agglomeration_level-1);

    directSolveCoarsest(dataBase, GAMGdata_, agglomeration_level);

    coarse2fine(dataBase, GAMGdata_, agglomeration_level, agglomeration_level-1, 1);

    fine2coarse(dataBase, GAMGdata_, agglomeration_level, 1, agglomeration_level-1);

    directSolveCoarsest(dataBase, GAMGdata_, agglomeration_level);

    coarse2fine(dataBase, GAMGdata_, agglomeration_level, agglomeration_level-1, 0);
};

void GAMGELLPreconditioner::precondition
(
    double *psi,
    const double *finestResidual,
    const dfMatrixDataBase& dataBase,
    GAMGStruct *GAMGdata_, int agglomeration_level
)
{

    std::cout << "******************************************************" << std::endl;
    std::cout << "********* call in GAMGELLPreconditioner::precondition " << std::endl;

    //TODO: get nVcycles from control files
    int nVcycles_ = 1; 
    initCycle(GAMGdata_, agglomeration_level);

    // Purpose: wA = 0.0;
    checkCudaErrors(cudaMemset(psi, 0, GAMGdata_[0].nCell*sizeof(double)));

    // Purpose: set GAMGdata_[0].d_Sources
    checkCudaErrors(cudaMemcpyAsync(GAMGdata_[0].d_Sources, finestResidual, GAMGdata_[0].nCell*sizeof(double), cudaMemcpyDeviceToDevice, dataBase.stream));

    for (int cycle=0; cycle<nVcycles_; cycle++)
    {
        // Purpose: do Vcycle calculation
        Vcycle(dataBase, GAMGdata_, agglomeration_level);

        // Purpose: use GAMGdata_[0].d_CorrFields to update psi
        updateCorrFieldGPU( dataBase.stream, GAMGdata_[0].nCell, psi, GAMGdata_[0].d_CorrFields);

        //add smoother for leveli=0, nFinestSweeps_
        //TODO: write nSweeps 
        smoother->smooth(dataBase.stream, nSweeps, GAMGdata_[0].nCell, psi, 
                    GAMGdata_[0].d_Sources, GAMGdata_[0].ell_row_maxcount, GAMGdata_[0].d_ell_cols,
                    GAMGdata_[0].d_ell_values, GAMGdata_[0].d_diag);


        if (cycle < nVcycles_-1)
        {
            // Purpose: Calculate finest level residual field to update finestResidual
            updateSourceFieldGPU_ell( dataBase, GAMGdata_[0].nCell, 
                                GAMGdata_[0].d_Sources, GAMGdata_[0].d_AcfField, psi,
                                GAMGdata_[0].d_diag, GAMGdata_[0].ell_row_maxcount, 
                                GAMGdata_[0].d_ell_cols, GAMGdata_[0].d_ell_values, 
                                GAMGdata_[0].d_interfaceIntCoeffs, GAMGdata_[0].d_interfaceBouCoeffs,
                                GAMGdata_[0].d_faceCells, GAMGdata_[0].nPatchFaces);
        }
    }
    std::cout << "********** end in GAMGELLPreconditioner::precondition " << std::endl;
    std::cout << "******************************************************" << std::endl;
};

