#include "dfELLSolver.H"
#include "dfSolverOpBase.H"
#include "dfMatrixDataBase.H"

// #define PARALLEL_
#define PRINT_

// kernel functions for PCG solver
void PCGELLSolver::initialize(const int nCells, const size_t boundary_surface_value_bytes)
{
    // cudamalloc variables related to PCGSolver
    cudaMalloc(&d_wA, nCells * sizeof(double));
    cudaMalloc(&d_rA, nCells * sizeof(double));
    cudaMalloc(&d_pA, nCells * sizeof(double));
    cudaMalloc(&d_normFactors_tmp, nCells * sizeof(double));
    cudaMalloc(&d_wArA_tmp, nCells * sizeof(double));
    cudaMalloc(&d_wApA_tmp, nCells * sizeof(double));
    cudaMalloc(&reduce_result, sizeof(double));
    // for parallel
    cudaMalloc(&scalarSendBufList_, boundary_surface_value_bytes);
    cudaMalloc(&scalarRecvBufList_, boundary_surface_value_bytes);

    // preconditioner

}

void PCGELLSolver::initializeStream(const int nCells, const size_t boundary_surface_value_bytes, cudaStream_t stream)
{
    // cudamalloc variables related to PCGSolver
    cudaMallocAsync(&d_wA, nCells * sizeof(double), stream);
    cudaMallocAsync(&d_rA, nCells * sizeof(double), stream);
    cudaMallocAsync(&d_pA, nCells * sizeof(double), stream);
    cudaMallocAsync(&d_normFactors_tmp, nCells * sizeof(double), stream);
    cudaMallocAsync(&d_wArA_tmp, nCells * sizeof(double), stream);
    cudaMallocAsync(&d_wApA_tmp, nCells * sizeof(double), stream);
    cudaMallocAsync(&reduce_result, sizeof(double), stream);
    // for parallel
    cudaMallocAsync(&scalarSendBufList_, boundary_surface_value_bytes, stream);
    cudaMallocAsync(&scalarRecvBufList_, boundary_surface_value_bytes, stream);

    // preconditioner

}

void PCGELLSolver::freeInit(){
    cudaFree(d_wA);
    cudaFree(d_rA);
    cudaFree(d_pA);
    cudaFree(d_normFactors_tmp);
    cudaFree(d_wArA_tmp);
    cudaFree(d_wApA_tmp);
    cudaFree(reduce_result);
    cudaFree(scalarSendBufList_);
    cudaFree(scalarRecvBufList_);
}

void PCGELLSolver::freeInitStream(cudaStream_t stream)
{
    cudaFreeAsync(d_wA, stream);
    cudaFreeAsync(d_rA, stream);
    cudaFreeAsync(d_pA, stream);
    cudaFreeAsync(d_normFactors_tmp, stream);
    cudaFreeAsync(d_wArA_tmp, stream);
    cudaFreeAsync(d_wApA_tmp, stream);
    cudaFreeAsync(reduce_result, stream);
    cudaFreeAsync(scalarSendBufList_, stream);
    cudaFreeAsync(scalarRecvBufList_, stream);
}

void PCGELLSolver::initializeGAMG(const int nCells, const size_t boundary_surface_value_bytes,
                    GAMGStruct *GAMGdata_, int agglomeration_level)
{
    // cudamalloc variables related to PCGSolver
    cudaMalloc(&d_wA, nCells * sizeof(double));
    cudaMalloc(&d_rA, nCells * sizeof(double));
    cudaMalloc(&d_pA, nCells * sizeof(double));
    cudaMalloc(&d_normFactors_tmp, nCells * sizeof(double));
    cudaMalloc(&d_wArA_tmp, nCells * sizeof(double));
    cudaMalloc(&d_wApA_tmp, nCells * sizeof(double));
    cudaMalloc(&reduce_result, sizeof(double));
    // for parallel
    cudaMalloc(&scalarSendBufList_, boundary_surface_value_bytes);
    cudaMalloc(&scalarRecvBufList_, boundary_surface_value_bytes);

    // preconditioner
    precond_ = new GAMGELLPreconditioner();
    precond_->initialize(GAMGdata_, agglomeration_level);
}

void PCGELLSolver::initGAMGMatrix(const dfMatrixDataBase& dataBase, GAMGStruct *GAMGdata_, int agglomeration_level)
{
    // preconditioner
    precond_->agglomerateMatrix(dataBase, GAMGdata_, agglomeration_level);
}

void PCGELLSolver::solve
(
        const dfMatrixDataBase& dataBase,
        const double* d_internal_coeffs,
        const double* d_boundary_coeffs,
        int* patch_type,
        double* diagPtr,
        double* ellValues,
        int* ellCols,
        int ell_max_count_,
        const double *rhs, 
        double *psi
)
{
    printf("GPU-ELL-PCG::solve start --------------------------------------------\n");

    int nIterations = 0;
 
    const int row_ = dataBase.num_total_cells;
    const int nCells = dataBase.num_cells;

    double wArA = 0.; // TODO: = solverPerf.great_
    double wArAold = wArA;

    // these two int control reduce's scale
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (nCells + threads_per_block - 1) / threads_per_block;

#ifdef PRINT_    
    printf("threads_per_block = %d, blocks_per_grid = %d\n",threads_per_block, blocks_per_grid);
#endif

    double psi_ave = 0.;
    double normFactor = 0.;
    double initialResidual = 0.;
    double finalResidual = 0.;

    // --- reduce psi to get : psi_ave ---
    reduce(nCells, threads_per_block, blocks_per_grid, psi, reduce_result, dataBase.stream, false);
#ifndef PARALLEL_
    cudaMemcpyAsync(&psi_ave, &reduce_result[0] , sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#else
    ncclAllReduce(&reduce_result[0], &reduce_result[0], 1, ncclDouble, ncclSum, dataBase.nccl_comm, dataBase.stream);
    cudaStreamSynchronize(dataBase.stream);
    cudaMemcpyAsync(&psi_ave, &reduce_result[0], sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#endif
    psi_ave = psi_ave / row_;

#ifdef PRINT_
    printf("psi_ave = %.10e\n",psi_ave);
#endif

    // --- addInternalCoeffs : diag ---
    // input : d_internal_coeffs
    addInternalCoeffs(dataBase.stream, dataBase.num_patches, dataBase.patch_size, 
        d_internal_coeffs, dataBase.d_boundary_face_cell, diagPtr, patch_type);
    
    // --- SpMV : wA ---
    // input : psi, diag
    SpMV4ELL(dataBase.stream, nCells, diagPtr, ellValues, ellCols, ell_max_count_, psi, d_wA);

#ifdef PARALLEL_
    // --- initMatrixInterfaces & updateMatrixInterfaces : wA ---
    // input : psi (neighbor's psi)
    updateMatrixInterfaces(
        dataBase.stream, dataBase.num_patches, dataBase.patch_size,
        dataBase.neighbProcNo, dataBase.nccl_comm,
        dataBase.interfaceFlag, psi, d_wA, 
        scalarSendBufList_, scalarRecvBufList_,
        d_boundary_coeffs, dataBase.d_boundary_face_cell, patch_type);
#endif

    // --- calculate : rA and pA ---
    // input : rhs, wA and diag
    calrAandpA4ELL(dataBase.stream, nCells, d_rA, rhs, d_wA, diagPtr, ellValues, ell_max_count_, d_pA);

        // --- subBoundaryCoeffs : pA ---
    // input : d_boundary_coeffs
    subBoundaryCoeffs(dataBase.stream, dataBase.num_patches, dataBase.patch_size,
        d_boundary_coeffs, dataBase.d_boundary_face_cell, d_pA, patch_type);

    // --- calculate : pA and d_normFactors_tmp ---
    // input : psi_ave and wA, pA, rhs
    calpAandnormFactor(dataBase.stream, nCells, psi_ave, d_pA, d_normFactors_tmp, d_wA, rhs);
    
    // --- reduce d_normFactors_tmp to get : normFactor ---
    reduce(nCells, threads_per_block, blocks_per_grid, d_normFactors_tmp, reduce_result, dataBase.stream, false);
#ifndef PARALLEL_
    cudaMemcpyAsync(&normFactor, &reduce_result[0] , sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#else
    ncclAllReduce(&reduce_result[0], &reduce_result[0], 1, ncclDouble, ncclSum, dataBase.nccl_comm, dataBase.stream);
    cudaStreamSynchronize(dataBase.stream);
    cudaMemcpyAsync(&normFactor, &reduce_result[0], sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#endif

    normFactor += small_;

#ifdef PRINT_
    printf("normFactor = %.10e\n",normFactor);
#endif

    // --- reduce abs(rA) to get : initialResidual ---
    reduce(nCells, threads_per_block, blocks_per_grid, d_rA, reduce_result, dataBase.stream, true);
#ifndef PARALLEL_
    cudaMemcpyAsync(&initialResidual, &reduce_result[0] , sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#else
    ncclAllReduce(&reduce_result[0], &reduce_result[0], 1, ncclDouble, ncclSum, dataBase.nccl_comm, dataBase.stream);
    cudaStreamSynchronize(dataBase.stream);
    cudaMemcpyAsync(&initialResidual, &reduce_result[0], sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#endif
        
    initialResidual = initialResidual / normFactor;

    finalResidual = initialResidual;

#ifdef PRINT_
    printf("first finalResidual = %.10e\n",finalResidual);
#endif

    if
    (
        minIter_ > 0
     || !checkConvergence(finalResidual, initialResidual, nIterations)
    ){

        do{

            wArAold = wArA;

            // TODO: precondition

            // --- calculate : d_wArA_tmp ---
            // input : wA, rA
            AmulBtoC(dataBase.stream, nCells, d_wA, d_rA, d_wArA_tmp);

            // --- reduce d_wArA_tmp to get : wArA ---
            reduce(nCells, threads_per_block, blocks_per_grid, d_wArA_tmp, reduce_result, dataBase.stream, false);
#ifndef PARALLEL_
            cudaMemcpyAsync(&wArA, &reduce_result[0] , sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#else
            ncclAllReduce(&reduce_result[0], &reduce_result[0], 1, ncclDouble, ncclSum, dataBase.nccl_comm, dataBase.stream);
            cudaStreamSynchronize(dataBase.stream);
            cudaMemcpyAsync(&wArA, &reduce_result[0], sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#endif

#ifdef PRINT_
            printf("wArA = %.10e\n",wArA);
#endif

            if(nIterations == 0){
                cudaMemcpyAsync(d_pA, d_wA, nCells * sizeof(double), cudaMemcpyDeviceToDevice, dataBase.stream);
            }
            else{
                double beta = wArA/wArAold;
                // --- calculate : d_pA ---
                // input : wA, beta, d_pA
                calpA(dataBase.stream, nCells, d_pA, d_wA, beta);
            }

            // --- SpMV : wA ---
            // input : pA, diag
            SpMV4ELL(dataBase.stream, nCells, diagPtr, ellValues, ellCols, ell_max_count_, d_pA, d_wA);

#ifdef PARALLEL_
            // --- initMatrixInterfaces & updateMatrixInterfaces wA ---
            // input : pA (neighbor's pA)
            updateMatrixInterfaces(
                dataBase.stream, dataBase.num_patches, dataBase.patch_size,
                dataBase.neighbProcNo, dataBase.nccl_comm,
                dataBase.interfaceFlag, d_pA, d_wA, 
                scalarSendBufList_, scalarRecvBufList_,
                d_boundary_coeffs, dataBase.d_boundary_face_cell, patch_type);
#endif

            double wApA = 0.;
            // --- calculate : d_wApA_tmp ---
            // input : wA, pA
            AmulBtoC(dataBase.stream, nCells, d_wA, d_pA, d_wApA_tmp);

            // --- reduce d_wApA_tmp to get : wApA ---
            reduce(nCells, threads_per_block, blocks_per_grid, d_wApA_tmp, reduce_result, dataBase.stream, false);
#ifndef PARALLEL_
            cudaMemcpyAsync(&wApA, &reduce_result[0] , sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#else
            ncclAllReduce(&reduce_result[0], &reduce_result[0], 1, ncclDouble, ncclSum, dataBase.nccl_comm, dataBase.stream);
            cudaStreamSynchronize(dataBase.stream);
            cudaMemcpyAsync(&wApA, &reduce_result[0], sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#endif

#ifdef PRINT_
            printf("wApA = %.10e\n",wApA);
#endif

            if (checkSingularity(abs(wApA)/normFactor)) break;

            double alpha = wArA/wApA;
            // --- calculate : psi and d_rA ---
            // input : alpha, d_pA and alpha, d_wA
            calpsiandrA(dataBase.stream, nCells, psi, d_pA, d_rA, d_wA, alpha);

            // --- reduce abs(rA) to get : finalResidual ---
            reduce(nCells, threads_per_block, blocks_per_grid, d_rA, reduce_result, dataBase.stream, true);
#ifndef PARALLEL_
            cudaMemcpyAsync(&finalResidual, &reduce_result[0] , sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#else
            ncclAllReduce(&reduce_result[0], &reduce_result[0], 1, ncclDouble, ncclSum, dataBase.nccl_comm, dataBase.stream);
            cudaStreamSynchronize(dataBase.stream);
            cudaMemcpyAsync(&finalResidual, &reduce_result[0], sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#endif

            finalResidual = finalResidual / normFactor;

#ifdef PRINT_
            printf("final finalResidual = finalResidual / normFactor : %.10e\n",finalResidual);
#endif
        }while
        (
            (
            ++nIterations < maxIter_
            && !checkConvergence(finalResidual, initialResidual, nIterations)
            )
            || nIterations < minIter_
        );

    }
};


void PCGELLSolver::solve_useGAMG
(
        const dfMatrixDataBase& dataBase,
        const double* d_internal_coeffs,
        const double* d_boundary_coeffs,
        int* patch_type,
        double* diagPtr,
        double* ellValues,
        int* ellCols,
        int ell_max_count_,
        const double *rhs, 
        double *psi,
        GAMGStruct *GAMGdata_, 
        int agglomeration_level
)
{
    printf("GPU-ELL-PCG(GAMG)::solve start --------------------------------------------\n");

    int nIterations = 0;
 
    const int row_ = dataBase.num_total_cells;
    const int nCells = dataBase.num_cells;

    double wArA = 0.; // TODO: = solverPerf.great_
    double wArAold = wArA;

    // these two int control reduce's scale
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (nCells + threads_per_block - 1) / threads_per_block;

#ifdef PRINT_    
    printf("threads_per_block = %d, blocks_per_grid = %d\n",threads_per_block, blocks_per_grid);
#endif

    double psi_ave = 0.;
    double normFactor = 0.;
    double initialResidual = 0.;
    double finalResidual = 0.;

    // --- reduce psi to get : psi_ave ---
    reduce(nCells, threads_per_block, blocks_per_grid, psi, reduce_result, dataBase.stream, false);
#ifndef PARALLEL_
    cudaMemcpyAsync(&psi_ave, &reduce_result[0] , sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#else
    ncclAllReduce(&reduce_result[0], &reduce_result[0], 1, ncclDouble, ncclSum, dataBase.nccl_comm, dataBase.stream);
    cudaStreamSynchronize(dataBase.stream);
    cudaMemcpyAsync(&psi_ave, &reduce_result[0], sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#endif
    psi_ave = psi_ave / row_;

#ifdef PRINT_
    printf("psi_ave = %.10e\n",psi_ave);
#endif

    // --- addInternalCoeffs : diag ---
    // input : d_internal_coeffs
    addInternalCoeffs(dataBase.stream, dataBase.num_patches, dataBase.patch_size, 
        d_internal_coeffs, dataBase.d_boundary_face_cell, diagPtr, patch_type);
    
    // --- SpMV : wA ---
    // input : psi, diag
    SpMV4ELL(dataBase.stream, nCells, diagPtr, ellValues, ellCols, ell_max_count_, psi, d_wA);

#ifdef PARALLEL_
    // --- initMatrixInterfaces & updateMatrixInterfaces : wA ---
    // input : psi (neighbor's psi)
    updateMatrixInterfaces(
        dataBase.stream, dataBase.num_patches, dataBase.patch_size,
        dataBase.neighbProcNo, dataBase.nccl_comm,
        dataBase.interfaceFlag, psi, d_wA, 
        scalarSendBufList_, scalarRecvBufList_,
        d_boundary_coeffs, dataBase.d_boundary_face_cell, patch_type);
#endif

    // --- calculate : rA and pA ---
    // input : rhs, wA and diag
    calrAandpA4ELL(dataBase.stream, nCells, d_rA, rhs, d_wA, diagPtr, ellValues, ell_max_count_, d_pA);

        // --- subBoundaryCoeffs : pA ---
    // input : d_boundary_coeffs
    subBoundaryCoeffs(dataBase.stream, dataBase.num_patches, dataBase.patch_size,
        d_boundary_coeffs, dataBase.d_boundary_face_cell, d_pA, patch_type);

    // --- calculate : pA and d_normFactors_tmp ---
    // input : psi_ave and wA, pA, rhs
    calpAandnormFactor(dataBase.stream, nCells, psi_ave, d_pA, d_normFactors_tmp, d_wA, rhs);
    
    // --- reduce d_normFactors_tmp to get : normFactor ---
    reduce(nCells, threads_per_block, blocks_per_grid, d_normFactors_tmp, reduce_result, dataBase.stream, false);
#ifndef PARALLEL_
    cudaMemcpyAsync(&normFactor, &reduce_result[0] , sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#else
    ncclAllReduce(&reduce_result[0], &reduce_result[0], 1, ncclDouble, ncclSum, dataBase.nccl_comm, dataBase.stream);
    cudaStreamSynchronize(dataBase.stream);
    cudaMemcpyAsync(&normFactor, &reduce_result[0], sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#endif

    normFactor += small_;

#ifdef PRINT_
    printf("normFactor = %.10e\n",normFactor);
#endif

    // --- reduce abs(rA) to get : initialResidual ---
    reduce(nCells, threads_per_block, blocks_per_grid, d_rA, reduce_result, dataBase.stream, true);
#ifndef PARALLEL_
    cudaMemcpyAsync(&initialResidual, &reduce_result[0] , sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#else
    ncclAllReduce(&reduce_result[0], &reduce_result[0], 1, ncclDouble, ncclSum, dataBase.nccl_comm, dataBase.stream);
    cudaStreamSynchronize(dataBase.stream);
    cudaMemcpyAsync(&initialResidual, &reduce_result[0], sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#endif
        
    initialResidual = initialResidual / normFactor;

    finalResidual = initialResidual;

#ifdef PRINT_
    printf("first finalResidual = %.10e\n",finalResidual);
#endif

    if
    (
        minIter_ > 0
     || !checkConvergence(finalResidual, initialResidual, nIterations)
    ){

        do{

            precond_->precondition(d_wA, d_rA, dataBase, GAMGdata_, agglomeration_level);

            wArAold = wArA;

            // --- calculate : d_wArA_tmp ---
            // input : wA, rA
            AmulBtoC(dataBase.stream, nCells, d_wA, d_rA, d_wArA_tmp);

            // --- reduce d_wArA_tmp to get : wArA ---
            reduce(nCells, threads_per_block, blocks_per_grid, d_wArA_tmp, reduce_result, dataBase.stream, false);
#ifndef PARALLEL_
            cudaMemcpyAsync(&wArA, &reduce_result[0] , sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#else
            ncclAllReduce(&reduce_result[0], &reduce_result[0], 1, ncclDouble, ncclSum, dataBase.nccl_comm, dataBase.stream);
            cudaStreamSynchronize(dataBase.stream);
            cudaMemcpyAsync(&wArA, &reduce_result[0], sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#endif

#ifdef PRINT_
            printf("wArA = %.10e\n",wArA);
#endif

            if(nIterations == 0){
                cudaMemcpyAsync(d_pA, d_wA, nCells * sizeof(double), cudaMemcpyDeviceToDevice, dataBase.stream);
            }
            else{
                double beta = wArA/wArAold;
                // --- calculate : d_pA ---
                // input : wA, beta, d_pA
                calpA(dataBase.stream, nCells, d_pA, d_wA, beta);
            }

            // --- SpMV : wA ---
            // input : pA, diag
            SpMV4ELL(dataBase.stream, nCells, diagPtr, ellValues, ellCols, ell_max_count_, d_pA, d_wA);

#ifdef PARALLEL_
            // --- initMatrixInterfaces & updateMatrixInterfaces wA ---
            // input : pA (neighbor's pA)
            updateMatrixInterfaces(
                dataBase.stream, dataBase.num_patches, dataBase.patch_size,
                dataBase.neighbProcNo, dataBase.nccl_comm,
                dataBase.interfaceFlag, d_pA, d_wA, 
                scalarSendBufList_, scalarRecvBufList_,
                d_boundary_coeffs, dataBase.d_boundary_face_cell, patch_type);
#endif

            double wApA = 0.;
            // --- calculate : d_wApA_tmp ---
            // input : wA, pA
            AmulBtoC(dataBase.stream, nCells, d_wA, d_pA, d_wApA_tmp);

            // --- reduce d_wApA_tmp to get : wApA ---
            reduce(nCells, threads_per_block, blocks_per_grid, d_wApA_tmp, reduce_result, dataBase.stream, false);
#ifndef PARALLEL_
            cudaMemcpyAsync(&wApA, &reduce_result[0] , sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#else
            ncclAllReduce(&reduce_result[0], &reduce_result[0], 1, ncclDouble, ncclSum, dataBase.nccl_comm, dataBase.stream);
            cudaStreamSynchronize(dataBase.stream);
            cudaMemcpyAsync(&wApA, &reduce_result[0], sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#endif

#ifdef PRINT_
            printf("wApA = %.10e\n",wApA);
#endif

            if (checkSingularity(abs(wApA)/normFactor)) break;

            double alpha = wArA/wApA;
            // --- calculate : psi and d_rA ---
            // input : alpha, d_pA and alpha, d_wA
            calpsiandrA(dataBase.stream, nCells, psi, d_pA, d_rA, d_wA, alpha);

            // --- reduce abs(rA) to get : finalResidual ---
            reduce(nCells, threads_per_block, blocks_per_grid, d_rA, reduce_result, dataBase.stream, true);
#ifndef PARALLEL_
            cudaMemcpyAsync(&finalResidual, &reduce_result[0] , sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#else
            ncclAllReduce(&reduce_result[0], &reduce_result[0], 1, ncclDouble, ncclSum, dataBase.nccl_comm, dataBase.stream);
            cudaStreamSynchronize(dataBase.stream);
            cudaMemcpyAsync(&finalResidual, &reduce_result[0], sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#endif

            finalResidual = finalResidual / normFactor;

#ifdef PRINT_
            printf("final finalResidual = finalResidual / normFactor : %.10e\n",finalResidual);
#endif
        }while
        (
            (
            ++nIterations < maxIter_
            && !checkConvergence(finalResidual, initialResidual, nIterations)
            )
            || nIterations < minIter_
        );

    }
};
