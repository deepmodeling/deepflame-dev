#pragma once

#include "dfBlockSubMatrix.H"
#include <memory>
#include <vector>
#include <tuple>

namespace Foam{

class dfDIASubMatrix : public dfBlockSubMatrix{
private:

    label distance_count_;
    std::unique_ptr<label[]> distance_list_;
    std::unique_ptr<scalar[]> values_;
    std::unique_ptr<label[]> value_ldu_idx_;

public:
    dfDIASubMatrix(label nRows, label nCols, const std::vector<std::tuple<label,label,scalar>>& rcvList);
    dfDIASubMatrix(label nRows, label nCols, const std::vector<std::tuple<label,label,label>>& rciList);
    ~dfDIASubMatrix(){}

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
