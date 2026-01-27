#pragma once
#include "scalar.H"
#include "label.H"
#include "messageStream.H"

namespace Foam{


class dfBlockSubMatrix{
protected:

    label nRows_;
    label nCols_;

public:
    dfBlockSubMatrix(label nRows, label nCols):nRows_(nRows),nCols_(nCols){}
    virtual ~dfBlockSubMatrix(){}

    label& nRows() { return nRows_; }
    label& nCols() { return nCols_; }

    const label& nRows() const { return nRows_; }
    const label& nCols() const { return nCols_; }

    virtual void valueCopyOffDiagBlock(const scalar* const __restrict__ lduValuePtr) = 0;
    virtual void valueCopyDiagBlock(const scalar* const __restrict__ lower, const scalar* const __restrict__ upper) = 0;

    virtual void SpMV(
        scalar* const __restrict__ ApsiPtr_offset,
        const scalar* const __restrict__ psiPtr_offset
    ) const = 0;

    virtual void SumA(
        scalar* const __restrict__ sumAPtr_offset
    ) const = 0;

    virtual void BsubApsi(
        scalar* const __restrict__ BPtr_offset,
        const scalar* const __restrict__ psiPtr_offset
    ) const = 0;

    virtual void GaussSeidel(
        scalar* const __restrict__ psiPtr_offset,
        const scalar* const __restrict__ BPtr_offset,
        const scalar* const __restrict__ diagPtr_offset
    ) const = 0;

    // virtual void Jacobi(
    //     scalar* const __restrict__ psiPtr_offset,
    //     const scalar* const __restrict__ psiOldPtr_offset,
    //     const scalar* const __restrict__ BPtr_offset,
    //     const scalar* const __restrict__ diagPtr_offset
    // ) const = 0;
};

}
