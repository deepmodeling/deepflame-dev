#include <nccl.h>
#include <cuda_runtime.h>
#include "dfELLSolver.H"
#include "dfMatrixDataBase.H"
#include <cmath>
#include "dfSolverOpBase.H"

#define PARALLEL_
#define PRINT_

/*------------------------------------------------malloc---------------------------------------------------------*/

void PBiCGStabELLSolver::initialize(const int nCells, const size_t boundary_surface_value_bytes)
{
    cudaMalloc(&d_yA, nCells * sizeof(double));
    cudaMalloc(&d_rA, nCells * sizeof(double));
    cudaMalloc(&d_pA, nCells * sizeof(double));
    cudaMalloc(&d_normFactors_tmp, nCells * sizeof(double));
    cudaMalloc(&d_AyA, nCells * sizeof(double));
    cudaMalloc(&d_sA, nCells * sizeof(double));
    cudaMalloc(&d_zA, nCells * sizeof(double));
    cudaMalloc(&d_tA, nCells * sizeof(double));
    cudaMalloc(&d_rA0, nCells * sizeof(double));
    cudaMalloc(&d_rA0rA_tmp, nCells * sizeof(double));
    cudaMalloc(&d_rA0AyA_tmp, nCells * sizeof(double));
    cudaMalloc(&d_tAtA_tmp, nCells * sizeof(double));
    cudaMalloc(&d_sAtA_tmp, nCells * sizeof(double));
    cudaMalloc(&reduce_result, sizeof(double));
    // for parallel
    cudaMalloc(&scalarSendBufList_, boundary_surface_value_bytes);
    cudaMalloc(&scalarRecvBufList_, boundary_surface_value_bytes);
}

void PBiCGStabELLSolver::initializeStream(const int nCells, const size_t boundary_surface_value_bytes, cudaStream_t stream)
{
    cudaMallocAsync(&d_yA, nCells * sizeof(double), stream);
    cudaMallocAsync(&d_rA, nCells * sizeof(double), stream);
    cudaMallocAsync(&d_pA, nCells * sizeof(double), stream);
    cudaMallocAsync(&d_normFactors_tmp, nCells * sizeof(double), stream);
    cudaMallocAsync(&d_AyA, nCells * sizeof(double), stream);
    cudaMallocAsync(&d_sA, nCells * sizeof(double), stream);
    cudaMallocAsync(&d_zA, nCells * sizeof(double), stream);
    cudaMallocAsync(&d_tA, nCells * sizeof(double), stream);
    cudaMallocAsync(&d_rA0, nCells * sizeof(double), stream);
    cudaMallocAsync(&d_rA0rA_tmp, nCells * sizeof(double), stream);
    cudaMallocAsync(&d_rA0AyA_tmp, nCells * sizeof(double), stream);
    cudaMallocAsync(&d_tAtA_tmp, nCells * sizeof(double), stream);
    cudaMallocAsync(&d_sAtA_tmp, nCells * sizeof(double), stream);
    cudaMallocAsync(&reduce_result, sizeof(double), stream);
    // for parallel
    cudaMallocAsync(&scalarSendBufList_, boundary_surface_value_bytes, stream);
    cudaMallocAsync(&scalarRecvBufList_, boundary_surface_value_bytes, stream);
}

void PBiCGStabELLSolver::freeInit(){
    cudaFree(d_yA);
    cudaFree(d_rA);
    cudaFree(d_pA);
    cudaFree(d_normFactors_tmp);
    cudaFree(d_AyA);
    cudaFree(d_sA);
    cudaFree(d_zA);
    cudaFree(d_tA);
    cudaFree(d_rA0);
    cudaFree(d_rA0rA_tmp);
    cudaFree(d_rA0AyA_tmp);
    cudaFree(d_tAtA_tmp);
    cudaFree(d_sAtA_tmp);
    cudaFree(reduce_result);
    cudaFree(scalarSendBufList_);
    cudaFree(scalarRecvBufList_);
}

void PBiCGStabELLSolver::freeInitStream(cudaStream_t stream){
    cudaFreeAsync(d_yA, stream);
    cudaFreeAsync(d_rA, stream);
    cudaFreeAsync(d_pA, stream);
    cudaFreeAsync(d_normFactors_tmp, stream);
    cudaFreeAsync(d_AyA, stream);
    cudaFreeAsync(d_sA, stream);
    cudaFreeAsync(d_zA, stream);
    cudaFreeAsync(d_tA, stream);
    cudaFreeAsync(d_rA0, stream);
    cudaFreeAsync(d_rA0rA_tmp, stream);
    cudaFreeAsync(d_rA0AyA_tmp, stream);
    cudaFreeAsync(d_tAtA_tmp, stream);
    cudaFreeAsync(d_sAtA_tmp, stream);
    cudaFreeAsync(reduce_result, stream);
    cudaFreeAsync(scalarSendBufList_, stream);
    cudaFreeAsync(scalarRecvBufList_, stream);
}

void PBiCGStabELLSolver::initGAMGMatrix(const dfMatrixDataBase& dataBase, GAMGStruct *GAMGdata_, int agglomeration_level)
{
    // preconditioner
    std::cout << "********* call in PBiCGStabELLSolver::initGAMGMatrix() " << std::endl;
}

void PBiCGStabELLSolver::initializeGAMG(const int nCells, const size_t boundary_surface_value_bytes,
            GAMGStruct *GAMGdata_, int agglomeration_level) {};
            
void PBiCGStabELLSolver::solve_useGAMG
(
    const dfMatrixDataBase& dataBase,
    const double* d_internal_coeffs,
    const double* d_boundary_coeffs,
    int* patch_type,
    double* diagPtr,
    double* d_ell_values,
    int* d_ell_cols,
    int ell_row_maxcount,
    const double *rhs, 
    double *psi,
    GAMGStruct *GAMGdata_, 
    int agglomeration_level
){};

/*------------------------------------------------solve---------------------------------------------------------*/

void PBiCGStabELLSolver::solve
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
    printf("GPU-ELL-PBiCGStab::solve start --------------------------------------------\n");

    int nIterations = 0;
 
    const int row_ = dataBase.num_total_cells;
    const int nCells = dataBase.num_cells;

    // these two int control reduce's scale
    size_t threads_per_block = 1024;
    size_t blocks_per_grid = (nCells + threads_per_block - 1) / threads_per_block;

#ifdef PRINT_
    printf("reduce scale : threads_per_block = %d, blocks_per_grid = %d\n",threads_per_block, blocks_per_grid);
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

    // --- SpMV : yA ---
    // input : psi, diag
    SpMV4ELL(dataBase.stream, nCells, diagPtr, ellValues, ellCols, ell_max_count_, psi, d_yA);

#ifdef PARALLEL_      
    // --- initMatrixInterfaces & updateMatrixInterfaces : yA ---
    // input : psi (neighbor's psi)
    updateMatrixInterfaces(
        dataBase.stream, dataBase.num_patches, dataBase.patch_size,
        dataBase.neighbProcNo, dataBase.nccl_comm,
        dataBase.interfaceFlag, psi, d_yA, 
        scalarSendBufList_, scalarRecvBufList_,
        d_boundary_coeffs, dataBase.d_boundary_face_cell, patch_type);
    
#endif

    // --- calculate : rA and pA ---
    // input : rhs, yA and diag
    calrAandpA4ELL(dataBase.stream, nCells, d_rA, rhs, d_yA, diagPtr, ellValues, ell_max_count_, d_pA);

    // --- subBoundaryCoeffs : pA ---
    // input : d_boundary_coeffs
    subBoundaryCoeffs(dataBase.stream, dataBase.num_patches, dataBase.patch_size,
        d_boundary_coeffs, dataBase.d_boundary_face_cell, d_pA, patch_type);

    // --- calculate : pA and d_normFactors_tmp ---
    // input : psi_ave and yA, pA, rhs
    calpAandnormFactor(dataBase.stream, nCells, psi_ave, d_pA, d_normFactors_tmp, d_yA, rhs);

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
        
        // --- Store initial residual : rA0 ---
        // input : rA
        cudaMemcpyAsync(d_rA0, d_rA, nCells * sizeof(double), cudaMemcpyDeviceToDevice, dataBase.stream);

        double rA0rA = 0.;
        double alpha = 0.;
        double omega = 0.;
        double rA0AyA = 0.;
        double tAtA = 0.;

        // --- Solver iteration ---
        do
        {

#ifdef PRINT_
            printf("nIterations = %d\n",nIterations);
#endif

            // --- Store previous : rA0rAold ---
            const double rA0rAold = rA0rA;

            // --- calculate : d_rA0rA_tmp ---
            // input : rA, rA0
            AmulBtoC(dataBase.stream, nCells, d_rA, d_rA0, d_rA0rA_tmp);
            
            // --- reduce d_rA0rA_tmp to get : rA0rA ---
            reduce(nCells, threads_per_block, blocks_per_grid, d_rA0rA_tmp, reduce_result, dataBase.stream, false);
#ifndef PARALLEL_
            cudaMemcpyAsync(&rA0rA, &reduce_result[0] , sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#else
            ncclAllReduce(&reduce_result[0], &reduce_result[0], 1, ncclDouble, ncclSum, dataBase.nccl_comm, dataBase.stream);
            cudaStreamSynchronize(dataBase.stream);
            cudaMemcpyAsync(&rA0rA, &reduce_result[0], sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#endif

#ifdef PRINT_
            printf("rA0rA = %.5e\n",rA0rA);
#endif

            if (checkSingularity(abs(rA0rA)))
            {
                break;
            }

            // --- Update pA

            if(nIterations == 0){
                // --- calculate pA and yA ---
                // input : rA and pA
                calpAandyAInit(dataBase.stream, nCells, d_pA, d_rA, d_yA);
            }else{
                if(checkSingularity(std::abs(omega))){
                    break;
                }
                // --- calculate pA and yA ---
                // input : rA, beta, pA, omega, AyA and pA
                double beta = (rA0rA/rA0rAold)*(alpha/omega);
                calpAandyA(dataBase.stream, nCells, d_pA, d_rA, beta, omega, d_AyA, d_yA);
            }

            // --- SpMV : AyA ---
            // input : yA, diag
            SpMV4ELL(dataBase.stream, nCells, diagPtr, ellValues, ellCols, ell_max_count_, d_yA, d_AyA);

#ifdef PARALLEL_
            // --- initMatrixInterfaces & updateMatrixInterfaces AyA ---
            // input : yA (neighbor's yA)
            updateMatrixInterfaces(
                dataBase.stream, dataBase.num_patches, dataBase.patch_size,
                dataBase.neighbProcNo, dataBase.nccl_comm,
                dataBase.interfaceFlag, d_yA, d_AyA, 
                scalarSendBufList_, scalarRecvBufList_,
                d_boundary_coeffs, dataBase.d_boundary_face_cell, patch_type);
#endif

            // --- calculate : d_rA0AyA_tmp ---
            // input : rA0, AyA
            AmulBtoC(dataBase.stream, nCells, d_rA0, d_AyA, d_rA0AyA_tmp);

            // --- reduce d_rA0AyA_tmp to get : rA0AyA ---
            reduce(nCells, threads_per_block, blocks_per_grid, d_rA0AyA_tmp, reduce_result, dataBase.stream, false);
#ifndef PARALLEL_
            cudaMemcpyAsync(&rA0AyA, &reduce_result[0] , sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#else
            ncclAllReduce(&reduce_result[0], &reduce_result[0], 1, ncclDouble, ncclSum, dataBase.nccl_comm, dataBase.stream);
            cudaStreamSynchronize(dataBase.stream);
            cudaMemcpyAsync(&rA0AyA, &reduce_result[0], sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#endif

            alpha = rA0rA/rA0AyA;
            
#ifdef PRINT_
            printf("alpha = rA0rA/rA0AyA : %.10e, %.10e, %.10e\n", alpha, rA0rA, rA0AyA);
#endif

            // --- calculate : sA ---
            // input : rA, alpha, AyA
            calsA(dataBase.stream,nCells, d_sA, d_rA, alpha, d_AyA);

            // --- reduce abs(sA) to get : finalResidual ---
            reduce(nCells, threads_per_block, blocks_per_grid, d_sA, reduce_result, dataBase.stream, true);
#ifndef PARALLEL_
            cudaMemcpyAsync(&finalResidual, &reduce_result[0] , sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#else
            ncclAllReduce(&reduce_result[0], &reduce_result[0], 1, ncclDouble, ncclSum, dataBase.nccl_comm, dataBase.stream);
            cudaStreamSynchronize(dataBase.stream);
            cudaMemcpyAsync(&finalResidual, &reduce_result[0], sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#endif

            finalResidual = finalResidual / normFactor;

#ifdef PRINT_
            printf("second finalResidual = finalResidual / normFactor : %.10e\n", finalResidual);
#endif
            
            if
            (
                checkConvergence(finalResidual, initialResidual, nIterations)
            )
            {
                // --- calculate : psi ---
                // input : alpha, yA
                exitLoop(dataBase.stream, nCells, alpha, d_yA, psi);

#ifdef PRINT_
                printf("exitLoop and mid!\n");
#endif
                
                nIterations++;

                break;
            }

            // --- Precondition : sA ---
            // input : zA
            cudaMemcpyAsync(d_zA, d_sA, nCells * sizeof(double), cudaMemcpyDeviceToDevice, dataBase.stream);

            // --- SpMV : tA ---
            // input : sA, diag
            SpMV4ELL(dataBase.stream, nCells, diagPtr, ellValues, ellCols, ell_max_count_, d_zA, d_tA);

#ifdef PARALLEL_
            // --- initMatrixInterfaces & updateMatrixInterfaces tA ---
            // input : zA (neighbor's zA)
            updateMatrixInterfaces(
                dataBase.stream, dataBase.num_patches, dataBase.patch_size,
                dataBase.neighbProcNo, dataBase.nccl_comm,
                dataBase.interfaceFlag, d_zA, d_tA, 
                scalarSendBufList_, scalarRecvBufList_,
                d_boundary_coeffs, dataBase.d_boundary_face_cell, patch_type);
#endif

            // --- calculate : d_tAtA_tmp ---
            // input : tA
            AmulAtoB(dataBase.stream, nCells, d_tA, d_tAtA_tmp);

            // --- reduce d_tAtA_tmp to get : tAtA ---
            reduce(nCells, threads_per_block, blocks_per_grid, d_tAtA_tmp, reduce_result, dataBase.stream, false);
#ifndef PARALLEL_
            cudaMemcpyAsync(&tAtA, &reduce_result[0] , sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#else
            ncclAllReduce(&reduce_result[0], &reduce_result[0], 1, ncclDouble, ncclSum, dataBase.nccl_comm, dataBase.stream);
            cudaStreamSynchronize(dataBase.stream);
            cudaMemcpyAsync(&tAtA, &reduce_result[0], sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#endif

            // --- calculate : d_sAtA_tmp ---
            // input : sA, tA
            AmulBtoC(dataBase.stream, nCells, d_sA, d_tA, d_sAtA_tmp);

            // --- reduce d_sAtA_tmp to get : omega ---
            reduce(nCells, threads_per_block, blocks_per_grid, d_sAtA_tmp, reduce_result, dataBase.stream, false);
#ifndef PARALLEL_
            cudaMemcpyAsync(&omega, &reduce_result[0] , sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#else
            ncclAllReduce(&reduce_result[0], &reduce_result[0], 1, ncclDouble, ncclSum, dataBase.nccl_comm, dataBase.stream);
            cudaStreamSynchronize(dataBase.stream);
            cudaMemcpyAsync(&omega, &reduce_result[0], sizeof(double), cudaMemcpyDeviceToHost, dataBase.stream);
#endif
            omega = omega / tAtA;

#ifdef PRINT_
            printf("omega = omega / tAtA : %.10e, %.10e\n", omega, tAtA);
#endif

            // --- calculate : psi and rA ---
            // input : alpha, yA, omega, zA and sA, omega, tA
            calpsiandrA(dataBase.stream, nCells, psi, d_yA, d_zA, d_rA, d_sA, d_tA, alpha, omega);

            // --- reduce d_rA to get : finalResidual ---
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
            printf("third finalResidual = finalResidual / normFactor : %.10e\n",finalResidual);
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

}
