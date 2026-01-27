#include "dfLduMatrix.H"
#include <cassert>

namespace Foam{

dfLduMatrix::dfLduMatrix(const lduMatrix& ldu):dfInnerMatrix(ldu){
    Info << "Building LduMatrix n_ : " << n_ << endl;
    lowerAddr_ = ldu.lduAddr().lowerAddr();
    upperAddr_ = ldu.lduAddr().upperAddr();
    ownerStartAddr_ = ldu.lduAddr().ownerStartAddr();
    losortAddr_ = ldu.lduAddr().losortAddr();
    if(ldu.hasLower())
    {
        lower_ = ldu.lower();
    }
    if(ldu.hasUpper())
    {
        upper_ = ldu.upper();
    }
}

dfLduMatrix::dfLduMatrix(const lduMesh& mesh):dfInnerMatrix(mesh){
    Info << "Building LduMatrix n_ : " << n_ << endl;
    lowerAddr_ = mesh.lduAddr().lowerAddr();
    upperAddr_ = mesh.lduAddr().upperAddr();
    ownerStartAddr_ = mesh.lduAddr().ownerStartAddr();
    losortAddr_ = mesh.lduAddr().losortAddr();
}

void dfLduMatrix::valueCopy(const lduMatrix& ldu){
    dfInnerMatrix::valueCopy(ldu);
    if(ldu.hasLower())
    {
        lower_ = ldu.lower();
    }
    if(ldu.hasUpper())
    {
        upper_ = ldu.upper();
    }
}

void dfLduMatrix::SpMV(scalar* const __restrict__ ApsiPtr, const scalar* const __restrict__ psiPtr) const {
    const label* const __restrict__ uPtr = upperAddr().begin();
    const label* const __restrict__ lPtr = lowerAddr().begin();

    const scalar* const __restrict__ diagPtr = diag().begin();
    const scalar* const __restrict__ upperPtr = upper().begin();
    const scalar* const __restrict__ lowerPtr = lower().begin();

    const label nCells = diag().size();
    for (label cell=0; cell<nCells; cell++)
    {
        ApsiPtr[cell] = diagPtr[cell]*psiPtr[cell];
    }

    const label nFaces = upper().size();

    for (label face=0; face<nFaces; face++)
    {
        ApsiPtr[uPtr[face]] += lowerPtr[face]*psiPtr[lPtr[face]];
        ApsiPtr[lPtr[face]] += upperPtr[face]*psiPtr[uPtr[face]];
    }
}

void dfLduMatrix::SumA(scalar* const __restrict__ sumAPtr) const {
    const label* const __restrict__ uPtr = upperAddr().begin();
    const label* const __restrict__ lPtr = lowerAddr().begin();

    const scalar* const __restrict__ diagPtr = diag().begin();
    const scalar* const __restrict__ upperPtr = upper().begin();
    const scalar* const __restrict__ lowerPtr = lower().begin();

    label nCells = diag().size();
    label nFaces = upper().size();

    for (label cell=0; cell<nCells; cell++)
    {
        sumAPtr[cell] = diagPtr[cell];
    }

    for (label face=0; face<nFaces; face++)
    {
        sumAPtr[uPtr[face]] += lowerPtr[face];
        sumAPtr[lPtr[face]] += upperPtr[face];
    }
}

void dfLduMatrix::GaussSeidel(scalar* const __restrict__ psiPtr, scalar* const __restrict__ bPrimePtr) const {
    const label nCells = diag().size();

    const scalar* const __restrict__ diagPtr = diag().begin();
    const scalar* const __restrict__ upperPtr = upper().begin();
    const scalar* const __restrict__ lowerPtr = lower().begin();
    const label* const __restrict__ uPtr = upperAddr().begin();
    const label* const __restrict__ ownStartPtr = ownerStartAddr().begin();

    for (label celli=0; celli<nCells; celli++)
    {   
        // Start and end of this row
        label fStart = ownStartPtr[celli];
        label fEnd = ownStartPtr[celli + 1];
        // Get the accumulated neighbour side
        scalar psii = bPrimePtr[celli];

        // Accumulate the owner product side
        for (label facei=fStart; facei<fEnd; facei++)
        {
            psii -= upperPtr[facei] * psiPtr[uPtr[facei]];
        }

        // Finish psi for this cell
        psii /= diagPtr[celli];

        // Distribute the neighbour side using psi for this cell
        for (label facei=fStart; facei<fEnd; facei++)
        {
            bPrimePtr[uPtr[facei]] -= lowerPtr[facei]*psii;
        }

        psiPtr[celli] = psii;
    }
}

void dfLduMatrix::Jacobi(scalar* const __restrict__ psiPtr, scalar* const __restrict__ bPrimePtr) const {
    const label nCells = diag().size();

    const scalar* const __restrict__ diagPtr = diag().begin();
    const scalar* const __restrict__ upperPtr = upper().begin();
    const scalar* const __restrict__ lowerPtr = lower().begin();
    const label* const __restrict__ uPtr = upperAddr().begin();
    const label* const __restrict__ ownStartPtr = ownerStartAddr().begin();

    for (label celli=0; celli<nCells; celli++)
    {   
        // Start and end of this row
        label fStart = ownStartPtr[celli];
        label fEnd = ownStartPtr[celli + 1];
        // Get the accumulated neighbour side

        // Distribute the neighbour side using psi for this cell
        for (label facei=fStart; facei<fEnd; facei++)
        {
            bPrimePtr[uPtr[facei]] -= lowerPtr[facei] * psiPtr[celli];
        }

        scalar psii = bPrimePtr[celli];

        // Accumulate the owner product side
        for (label facei=fStart; facei<fEnd; facei++)
        {
            psii -= upperPtr[facei] * psiPtr[uPtr[facei]];
        }

        // Finish psi for this cell
        psii /= diagPtr[celli];

        psiPtr[celli] = psii;
    }
}

void dfLduMatrix::calcDILUReciprocalD(scalar* const __restrict__ rDPtr) const {
    assert(this->asymmetric());

    const label* const __restrict__ uPtr = upperAddr().begin();
    const label* const __restrict__ lPtr = lowerAddr().begin();

    const scalar* const __restrict__ upperPtr = upper().begin();
    const scalar* const __restrict__ lowerPtr = lower().begin();

    label nFaces = upper().size();

    for (label face=0; face<nFaces; face++)
    {
        rDPtr[uPtr[face]] -= upperPtr[face]*lowerPtr[face]/rDPtr[lPtr[face]];
    }

    // Calculate the reciprocal of the preconditioned diagonal
    label nCells = diag().size();

    for (label cell=0; cell<nCells; cell++)
    {
        rDPtr[cell] = 1.0/rDPtr[cell];
    }
}

void dfLduMatrix::DILUPrecondition(scalar* const __restrict__ wAPtr, const scalar* const __restrict__ rAPtr, const scalar* const __restrict__ rDPtr) const {
    assert(this->asymmetric());
    const label* const __restrict__ uPtr = upperAddr().begin();
    const label* const __restrict__ lPtr = lowerAddr().begin();
    const label* const __restrict__ losortPtr = losortAddr().begin();

    const scalar* const __restrict__ upperPtr = upper().begin();
    const scalar* const __restrict__ lowerPtr = lower().begin();

    label nCells = diag().size();
    label nFaces = upper().size();
    label nFacesM1 = nFaces - 1;

    for (label cell=0; cell<nCells; cell++)
    {
        wAPtr[cell] = rDPtr[cell]*rAPtr[cell];
    }

    label sface;

    for (label face=0; face<nFaces; face++)
    {
        sface = losortPtr[face];
        wAPtr[uPtr[sface]] -=
            rDPtr[uPtr[sface]]*lowerPtr[sface]*wAPtr[lPtr[sface]];
    }

    for (label face=nFacesM1; face>=0; face--)
    {
        wAPtr[lPtr[face]] -=
            rDPtr[lPtr[face]]*upperPtr[face]*wAPtr[uPtr[face]];
    }
}

void dfLduMatrix::DILUPreconditionT(scalar* const __restrict__ wTPtr, const scalar* const __restrict__ rTPtr, const scalar* const __restrict__ rDPtr) const{
    assert(this->asymmetric());

    const label* const __restrict__ uPtr = upperAddr().begin();
    const label* const __restrict__ lPtr = lowerAddr().begin();
    const label* const __restrict__ losortPtr = losortAddr().begin();

    const scalar* const __restrict__ upperPtr = upper().begin();
    const scalar* const __restrict__ lowerPtr = lower().begin();

    label nCells = diag().size();
    label nFaces = upper().size();
    label nFacesM1 = nFaces - 1;

    for (label cell=0; cell<nCells; cell++)
    {
        wTPtr[cell] = rDPtr[cell]*rTPtr[cell];
    }

    for (label face=0; face<nFaces; face++)
    {
        wTPtr[uPtr[face]] -=
            rDPtr[uPtr[face]]*upperPtr[face]*wTPtr[lPtr[face]];
    }

    label sface;

    for (label face=nFacesM1; face>=0; face--)
    {
        sface = losortPtr[face];
        wTPtr[lPtr[sface]] -=
            rDPtr[lPtr[sface]]*lowerPtr[sface]*wTPtr[uPtr[sface]];
    }
}

void dfLduMatrix::calcDICReciprocalD(scalar* const __restrict__ rDPtr) const {
    assert(this->symmetric());

    const label* const __restrict__ uPtr = upperAddr().begin();
    const label* const __restrict__ lPtr = lowerAddr().begin();
    const scalar* const __restrict__ upperPtr = upper().begin();

    // Calculate the DIC diagonal
    const label nFaces = upper().size();
    for (label face=0; face<nFaces; face++)
    {
        rDPtr[uPtr[face]] -= upperPtr[face]*upperPtr[face]/rDPtr[lPtr[face]];
    }

    // Calculate the reciprocal of the preconditioned diagonal
    const label nCells = diag().size();

    for (label cell=0; cell<nCells; cell++)
    {
        rDPtr[cell] = 1.0/rDPtr[cell];
    }
}

void dfLduMatrix::DICPrecondition(scalar* const __restrict__ wAPtr, const scalar* const __restrict__ rAPtr, const scalar* const __restrict__ rDPtr) const {
    assert(this->symmetric());

    const label* const __restrict__ uPtr = upperAddr().begin();
    const label* const __restrict__ lPtr = lowerAddr().begin();
    const scalar* const __restrict__ upperPtr = upper().begin();

    label nCells = diag().size();
    label nFaces = upper().size();
    label nFacesM1 = nFaces - 1;

    for (label cell=0; cell<nCells; cell++)
    {
        wAPtr[cell] = rDPtr[cell]*rAPtr[cell];
    }

    for (label face=0; face<nFaces; face++)
    {
        wAPtr[uPtr[face]] -= rDPtr[uPtr[face]]*upperPtr[face]*wAPtr[lPtr[face]];
    }

    for (label face=nFacesM1; face>=0; face--)
    {
        wAPtr[lPtr[face]] -= rDPtr[lPtr[face]]*upperPtr[face]*wAPtr[uPtr[face]];
    }
}

} // End namespace Foam
