#pragma once
#include "dfInnerMatrix.H"


namespace Foam{

class dfLduMatrix : public dfInnerMatrix{
private:
    // - Coefficients (not including interfaces)
    scalarField lower_, upper_;
    labelList lowerAddr_;
    labelList upperAddr_;
    labelList ownerStartAddr_;
    labelList losortAddr_;

    scalarField& lower() {return symmetric_ ? upper_ : lower_;};
    scalarField& upper() {return upper_;};
    const scalarField& lower() const {return symmetric_ ? upper_ : lower_;};
    const scalarField& upper() const {return upper_;};
    labelList& lowerAddr() {return lowerAddr_;};
    labelList& upperAddr() {return upperAddr_;};
    const labelList& lowerAddr() const {return lowerAddr_;};
    const labelList& upperAddr() const {return upperAddr_;};
    labelList& ownerStartAddr() {return ownerStartAddr_;};
    const labelList& ownerStartAddr() const {return ownerStartAddr_;};
    labelList& losortAddr() {return losortAddr_;};
    const labelList& losortAddr() const {return losortAddr_;};

public:
    dfLduMatrix(const lduMatrix& ldu);

    dfLduMatrix(const lduMesh& mesh);

    void valueCopy(const lduMatrix& ldu);

    ~dfLduMatrix(){};

    InnerMatrixFormat getFormat() const {return InnerMatrixFormat::DFMATRIX_INNERMATRIX_FORMAT_LDU;};
    const std::string getFormatName() const {return "LDU";};

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