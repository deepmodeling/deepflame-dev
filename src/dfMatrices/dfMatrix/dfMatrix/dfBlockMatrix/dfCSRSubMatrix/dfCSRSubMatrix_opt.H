#pragma once

#include "dfBlockSubMatrix.H"
#include <memory>
#include <vector>
#include <tuple>
//#define CSR_SUB_USE_HBM

namespace Foam{

class dfCSRSubMatrix : public dfBlockSubMatrix{
private:

    std::unique_ptr<label[]> rowPtr_;
    
//#ifdef CSR_SUB_USE_HBM
    label * colIdx_, * value_ldu_idx_;
    scalar * values_;
//#else
//    std::unique_ptr<label[]> colIdx_;
//    std::unique_ptr<scalar[]> values_;
//    std::unique_ptr<label[]> value_ldu_idx_;
//#endif

public:
    dfCSRSubMatrix(label nRows, label nCols, const std::vector<std::tuple<label,label,scalar>>& rcvList);
    dfCSRSubMatrix(label nRows, label nCols, const std::vector<std::tuple<label,label,label>>& rciList);
    // dfCSRSubMatrix(label nRows, label nCols, label* rowPtr, label* colIdx, scalar* values):dfBlockSubMatrix(nRows, nCols),rowPtr_(rowPtr),colIdx_(colIdx),values_(values),value_ldu_idx_(nullptr){}
    // dfCSRSubMatrix(label nRows, label nCols, label* rowPtr, label* colIdx, scalar* values, label* value_ldu_idx):dfBlockSubMatrix(nRows, nCols),rowPtr_(rowPtr),colIdx_(colIdx),values_(values),value_ldu_idx_(value_ldu_idx){}
    ~dfCSRSubMatrix(){}

    void valueCopyOffDiagBlock(const scalar* const __restrict__ lduValuePtr);
    void valueCopyDiagBlock(const scalar* const __restrict__ lower, const scalar* const __restrict__ upper);

    void SpMV(
        scalar* const __restrict__ ApsiPtr_offset,
        const scalar* const __restrict__ psiPtr_offset
    ) const;

    void SumA(
        scalar* const __restrict__ sumAPtr_offset
    ) const;

    void BsubApsi(
        scalar* const __restrict__ BPtr_offset,
        const scalar* const __restrict__ psiPtr_offset
    ) const;

    void GaussSeidel(
        scalar* const __restrict__ psiPtr_offset,
        const scalar* const __restrict__ BPtr_offset,
        const scalar* const __restrict__ diagPtr_offset
    ) const;

    // void Jacobi(
    //     scalar* const __restrict__ psiPtr_offset,
    //     const scalar* const __restrict__ psiOldPtr_offset,
    //     const scalar* const __restrict__ BPtr_offset,
    //     const scalar* const __restrict__ diagPtr_offset
    // ) const;
};

}
