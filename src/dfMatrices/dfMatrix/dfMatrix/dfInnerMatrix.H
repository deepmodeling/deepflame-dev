#pragma once
#include "lduMatrix.H"
#include <cassert>
#include "messageStream.H"

namespace Foam{

enum InnerMatrixFormat{
    DFMATRIX_INNERMATRIX_FORMAT_LDU,
    DFMATRIX_INNERMATRIX_FORMAT_CSR,
    DFMATRIX_INNERMATRIX_FORMAT_BLOCK_CSR,
    DFMATRIX_INNERMATRIX_FORMAT_BLOCK_DIA,
};

class dfInnerMatrix{
protected:
    label n_;
    scalarField diag_;
    bool diagonal_;
    bool symmetric_;
public:
    dfInnerMatrix(const lduMatrix& ldu):
        n_(ldu.lduAddr().size()),
        diagonal_(ldu.diagonal()),
        symmetric_(ldu.symmetric())
    {
        assert(ldu.hasDiag());
        diag_ = ldu.diag();
    };

    dfInnerMatrix(const lduMesh& mesh):n_(mesh.lduAddr().size()){
        diag_.resize(n_);
    };

    dfInnerMatrix(const dfInnerMatrix& other) = delete;

    virtual void valueCopy(const lduMatrix& ldu){
        const auto& lduDiag = ldu.diag();
        diagonal_ = ldu.diagonal();
        symmetric_ = ldu.symmetric();
        #pragma omp parallel for 
        for(label i = 0; i < n_; ++i){
            diag_[i] = lduDiag[i];
        }
    }

    virtual ~dfInnerMatrix(){};

    const label& n() const {return n_;};
    scalarField& diag(){return diag_;};
    const scalarField& diag() const {return diag_;};
    bool diagonal() const {return diagonal_;};
    bool symmetric() const {return symmetric_;};
    bool asymmetric() const {return !symmetric_;};

    virtual InnerMatrixFormat getFormat() const = 0;
    virtual const std::string getFormatName() const = 0;

    virtual void SpMV
    (
        scalar* const __restrict__ ApsiPtr,
        const scalar* const __restrict__ psiPtr
    ) const = 0;
    virtual void SumA(
        scalar* const __restrict__ sumAPtr
    ) const = 0;
    virtual void GaussSeidel
    (
        scalar* const __restrict__ psiPtr,
        scalar* const __restrict__ bPrimePtr
    ) const = 0;
    virtual void Jacobi
    (
        scalar* const __restrict__ psiPtr,
        scalar* const __restrict__ bPrimePtr
    ) const = 0;
    virtual void calcDILUReciprocalD(
        scalar* const __restrict__ rDPtr
    ) const = 0;
    virtual void DILUPrecondition(
        scalar* const __restrict__ wAPtr,
        const scalar* const __restrict__ rAPtr,
        const scalar* const __restrict__ rDPtr
    ) const = 0;
    virtual void DILUPreconditionT(
        scalar* const __restrict__ wTPtr,
        const scalar* const __restrict__ rTPtr,
        const scalar* const __restrict__ rDPtr
    ) const = 0;
    virtual void calcDICReciprocalD(
        scalar* const __restrict__ rDPtr
    ) const = 0;
    virtual void DICPrecondition(
        scalar* const __restrict__ wAPtr,
        const scalar* const __restrict__ rAPtr,
        const scalar* const __restrict__ rDPtr
    ) const = 0;
};

} // End namespace Foam

