#pragma once
#include "dfInnerMatrix.H"
#include <vector>


namespace Foam{

class dfCSRMatrix : public dfInnerMatrix{
private:

    labelList rowPtr_;
    labelList colIdx_;
    scalarList values_;

    labelList value_ldu_idx_;

    const labelList& rowPtr() const {return rowPtr_;};
    const labelList& colIdx() const {return colIdx_;};
    const scalarList& values() const {return values_;};

    labelList& rowPtr() {return rowPtr_;};
    labelList& colIdx() {return colIdx_;};
    scalarList& values() {return values_;};

public:
    dfCSRMatrix(const lduMatrix& ldu);

    dfCSRMatrix(const lduMesh& mesh);

    void valueCopy(const lduMatrix& ldu);

    ~dfCSRMatrix(){};

    InnerMatrixFormat getFormat() const {return InnerMatrixFormat::DFMATRIX_INNERMATRIX_FORMAT_CSR;};
    const std::string getFormatName() const {return "CSR";};

    void SpMV(scalar* const __restrict__ ApsiPtr, const scalar* const __restrict__ psiPtr) const;
    
    void SumA(scalar* const __restrict__ sumPtr) const;

    void GaussSeidel(scalar* const __restrict__ psiPtr, scalar* const __restrict__ bPrimePtr) const;
    void Jacobi(scalar* const __restrict__ psiPtr, scalar* const __restrict__ bPrimePtr) const;

    void calcDILUReciprocalD(scalar* const __restrict__ rDPtr) const;
    void DILUPrecondition(scalar* const __restrict__ wAPtr, const scalar* const __restrict__ rAPtr, const scalar* const __restrict__ rDPtr) const;
    void DILUPreconditionT(scalar* const __restrict__ wTPtr, const scalar* const __restrict__ rTPtr, const scalar* const __restrict__ rDPtr) const;

    void calcDICReciprocalD(scalar* const __restrict__ rDPtr) const;
    void DICPrecondition(scalar* const __restrict__ wAPtr, const scalar* const __restrict__ rAPtr, const scalar* const __restrict__ rDPtr) const;
    
};

}