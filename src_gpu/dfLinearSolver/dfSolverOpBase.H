#pragma once
#include <cuda_runtime.h>
#include <nccl.h>
#include <vector>

using std::min;
using std::max;
using std::abs;

bool isPow2(int n);

void addInternalCoeffs(
        cudaStream_t stream, int num_patches, std::vector<int> patch_size, 
        const double *d_internal_coeffs, int *d_boundary_face_cell, 
        double *diagPtr, int *patch_type);

void SpMV4CSR(
        cudaStream_t stream, const int nCells, double *diagPtr, const double *off_diag_value,
        int *d_csr_row_index_no_diag, int *d_csr_col_index_no_diag, double *input, double *output);

void SpMV4ELL(
        cudaStream_t stream, const int nCells, double *diagPtr, double *ellValues, 
        int *ellCols, int ell_max_count_, double *input, double *output);

void calrAandpA4CSR(
        cudaStream_t stream, const int nCells, double *d_rA, const double *rhs, double *d_yA,
        double *diagPtr, const double *off_diag_value, int *d_csr_row_index_no_diag, double *d_pA);

void calrAandpA4ELL(
        cudaStream_t stream, const int nCells, double *d_rA, const double *rhs, 
        double *d_yA, double *diagPtr, double *ellValues, int ell_max_count_, double *d_pA);

void subBoundaryCoeffs(
        cudaStream_t stream, int num_patches, std::vector<int> patch_size, 
        const double *d_boundary_coeffs, int *d_boundary_face_cell, 
        double *d_pA, int *patch_type);

void calpAandnormFactor(
        cudaStream_t stream,int nCells, double psi_ave, double* pAPtr,
        double* normFactor, double* yAPtr, const double* source);

void AmulBtoC(
        cudaStream_t stream, int nCells, double *input1, double *input2, double *output);

void calpAandyAInit(
        cudaStream_t stream, int nCells, double *d_pA, double *d_rA, double *d_yA);

void calsA(
    cudaStream_t stream, int nCells, double *d_sA, 
    double *d_rA, double alpha, double *d_AyA);

void calpAandyA(
        cudaStream_t stream, int nCells, double *d_pA, double *d_rA,
        double beta, double omega, double *d_AyA, double *d_yA);

void exitLoop(
        cudaStream_t stream, int nCells, double input1, double *input2, double* output);

void AmulAtoB(
    cudaStream_t stream, int nCells, double *input, double* output);

void calpsiandrA(
    cudaStream_t stream, int nCells, double *psi, double *d_yA, double *d_zA, 
    double *d_rA, double *d_sA, double *d_tA, double alpha, double omega);

void updateMatrixInterfaces(
    cudaStream_t stream, int num_patches, std::vector<int> patch_size,
    std::vector<int> neighbProcNo,  ncclComm_t nccl_comm,
    int *interfaceFlag, double *input, double *output, 
    double *scalarSendBufList_, double *scalarRecvBufList_,
    const double *d_boundary_coeffs, int *d_boundary_face_cell, int *patch_type
);

// PCG
void calpA(
    cudaStream_t stream, int nCells, double* pAPtr, double* wAPtr, double beta
);

void calpsiandrA(
    cudaStream_t stream,  int nCells, double* psi, double* pA, double* rA, double* wA, double alpha
);

void reduce(int size, int threads, int blocks, double *d_idata, double *d_odata, cudaStream_t stream, const bool isabs);

// GAMG
void restrictFieldGPU(cudaStream_t stream, int nFineCells, int* d_restrictMap, 
                        double* d_fineField, double* d_coarseField);
                        
void restrictMatrixGPU(cudaStream_t stream, int nFineFaces, int* d_faceRestrictMap, int* d_faceFlipMap,
                        double* d_fineUpper, double* d_fineLower, 
                        double* d_coarseUpper, double* d_coarseLower, double* d_coarseDiag);

void prolongFieldGPU(cudaStream_t stream, int nFineCells, int* d_restrictMap, 
                        double* d_fineField, double* d_coarseField);

void scaleFieldGPU( cudaStream_t stream, ncclComm_t nccl_comm,int nCells, 
                    double* reduce_result,
                    double* d_Field, double* d_Source, 
                    double* d_AcfField, double* d_diag,
                    double* d_scalingFactorNum, double* d_scalingFactorDenom );

void updateSourceFieldGPU(cudaStream_t stream, int nCells, 
                        double* d_Field, double* d_AcfField);

void updateCorrFieldGPU(cudaStream_t stream, int nCells, 
                        double* d_Field, double* d_preSmoothField);

void directSolve1x1GPU(cudaStream_t stream, int nCells, 
                        double* d_diag, double* d_corrField, double* d_sourceField);