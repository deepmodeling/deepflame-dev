#include "dfCSRMatrix.H"
#include <cassert>
#include <mpi.h>

namespace Foam{

dfCSRMatrix::dfCSRMatrix(const lduMatrix& ldu):dfInnerMatrix(ldu){
    // Pout << "Enter dfCSRMatrix::dfCSRMatrix(lduMatrix& ldu)" << endl << flush;
    Info << "Building CSRMatrix n_ : " << n_ << endl;
    this->rowPtr_.resize(n_ + 1);
    
    // off_diagonal_nnz_ = 0;
    if(!ldu.hasLower() && !ldu.hasUpper()){
        for(label i = 0; i < n_ + 1; ++i){
            rowPtr_[i] = 0;
        }
        return;
    }

    const labelList& lduLowerAddr = ldu.lduAddr().lowerAddr();
    const labelList& lduUpperAddr = ldu.lduAddr().upperAddr();
    // lower[i] (lduUpperAddr[i], lduLowerAddr[i]) 
    const scalarList& lduLower = ldu.lower();
    // upper[i] (lduLowerAddr[i], lduUpperAddr[i])
    const scalarList& lduUpper = ldu.upper();
    
    // const labelUList& owner = mesh.owner();
    // const labelUList& neighbour = mesh.neighbour();
    label off_diagonal_nnz_ = lduLower.size() + lduUpper.size();

    this->colIdx_.resize(off_diagonal_nnz_);
    this->values_.resize(off_diagonal_nnz_);

    // compute row_count
    std::vector<label> row_count(n_, 0);
    std::vector<label> current_index(n_ + 1);

    assert(lduLower.size() == lduLowerAddr.size());
    for(label i = 0; i < lduLower.size(); ++i){
        label row = lduUpperAddr[i];
        row_count[row] += 1;
    }
    assert(lduUpper.size() == lduUpperAddr.size());
    for(label i = 0; i < lduUpper.size(); ++i){
        label row = lduLowerAddr[i];
        row_count[row] += 1;
    }

    rowPtr_[0] = 0;
    current_index[0] = 0;
    for(label i = 0; i < n_; ++i){
        rowPtr_[i + 1] = rowPtr_[i] + row_count[i];
        current_index[i + 1] = rowPtr_[i] + row_count[i];
    }

    // lower
    // (lduUpperAddr[i], lduLowerAddr[i])
    for(label i = 0; i < lduLower.size(); ++i){
        label r = lduUpperAddr[i];
        label c = lduLowerAddr[i];
        label index = current_index[r];
        // face2lower_[i] = index;
        colIdx_[index] = c;
        values_[index] = lduLower[i];
        current_index[r] += 1;
    }

    // for(label rc = 0; rc < n_; ++rc){
    //     label index = current_index[rc];
    //     colidx_[index] = rc;
    //     current_index[rc] += 1;
    // }

    // upper
    // (lduLowerAddr[i], lduUpperAddr[i])
    for(label i = 0; i < lduUpper.size(); ++i){
        label r = lduLowerAddr[i];
        label c = lduUpperAddr[i];
        label index = current_index[r];
        // face2upper_[i] = index;
        colIdx_[index] = c;
        values_[index] = lduUpper[i];
        current_index[r] += 1;
    }
    // Pout << "Exit dfCSRMatrix::dfCSRMatrix(lduMatrix& ldu)" << endl << flush;
}

dfCSRMatrix::dfCSRMatrix(const lduMesh& mesh):dfInnerMatrix(mesh){
    // Pout << "Enter dfCSRMatrix::dfCSRMatrix(lduMesh& mesh)" << endl << flush;
    Info << "Building CSRMatrix n_ : " << n_ << endl;
    this->rowPtr_.resize(n_ + 1);
    
    const labelList& lduLowerAddr = mesh.lduAddr().lowerAddr();
    const labelList& lduUpperAddr = mesh.lduAddr().upperAddr();

    assert(lduLowerAddr.size() == lduUpperAddr.size());

    label nFaces = lduUpperAddr.size();
    
    // lower[i] (lduUpperAddr[i], lduLowerAddr[i]) 
    // const scalarList& lduLower = ldu.lower();
    // upper[i] (lduLowerAddr[i], lduUpperAddr[i])
    // const scalarList& lduUpper = ldu.upper();
    
    // const labelUList& owner = mesh.owner();
    // const labelUList& neighbour = mesh.neighbour();
    label off_diagonal_nnz_ = 2 * nFaces;

    this->colIdx_.resize(off_diagonal_nnz_);
    this->values_.resize(off_diagonal_nnz_);
    this->value_ldu_idx_.resize(off_diagonal_nnz_);

    // compute row_count
    std::vector<label> row_count(n_, 0);
    std::vector<label> current_index(n_ + 1);

    // assert(lduLower.size() == lduLowerAddr.size());
    for(label i = 0; i < nFaces; ++i){
        label row = lduUpperAddr[i];
        row_count[row] += 1;
    }
    // assert(lduUpper.size() == lduUpperAddr.size());
    for(label i = 0; i < nFaces; ++i){
        label row = lduLowerAddr[i];
        row_count[row] += 1;
    }

    rowPtr_[0] = 0;
    current_index[0] = 0;
    for(label i = 0; i < n_; ++i){
        rowPtr_[i + 1] = rowPtr_[i] + row_count[i];
        current_index[i + 1] = rowPtr_[i] + row_count[i];
    }

    // lower
    // (lduUpperAddr[i], lduLowerAddr[i])
    for(label i = 0; i < nFaces; ++i){
        label r = lduUpperAddr[i];
        label c = lduLowerAddr[i];
        label index = current_index[r];
        colIdx_[index] = c;
        // values_[index] = lduLower[i];
        value_ldu_idx_[index] = i;
        current_index[r] += 1;
    }

    // for(label rc = 0; rc < n_; ++rc){
    //     label index = current_index[rc];
    //     colidx_[index] = rc;
    //     current_index[rc] += 1;
    // }

    // upper
    // (lduLowerAddr[i], lduUpperAddr[i])
    for(label i = 0; i < nFaces; ++i){
        label r = lduLowerAddr[i];
        label c = lduUpperAddr[i];
        label index = current_index[r];
        colIdx_[index] = c;
        // values_[index] = lduUpper[i];
        value_ldu_idx_[index] = i;
        current_index[r] += 1;
    }
    // Pout << "Exit dfCSRMatrix::dfCSRMatrix(lduMesh& mesh)" << endl << flush;
}

void dfCSRMatrix::valueCopy(const lduMatrix& ldu){
    dfInnerMatrix::valueCopy(ldu);

    const scalarList& lduLower = ldu.lower();
    const scalarList& lduUpper = ldu.upper();

    #pragma omp parallel for
    for(label r = 0; r < n_; ++r){
        for(label idx = rowPtr_[r]; idx < rowPtr_[r + 1]; ++idx){
            label c = colIdx_[idx];
            if(r > c){
                values_[idx] = lduLower[value_ldu_idx_[idx]];
            }else if(r < c){
                values_[idx] = lduUpper[value_ldu_idx_[idx]];
            }else{
                assert(false);
            }
        }
    }
}

void dfCSRMatrix::SpMV(scalar* const __restrict__ ApsiPtr, const scalar* const __restrict__ psiPtr) const {
    // Pout << "Enter dfCSRMatrix::SpMV(scalar* const __restrict__ ApsiPtr, const scalar* const __restrict__ psiPtr)" << endl << flush;
    const scalar* const __restrict__ diagPtr = diag().begin(); 
    
    #pragma omp parallel for
    for(label r = 0; r < n_; ++r){
        scalar sum = diagPtr[r] * psiPtr[r];
        for(label j = rowPtr_[r]; j < rowPtr_[r + 1]; ++j){
            sum += values_[j] * psiPtr[colIdx_[j]];
        }
        ApsiPtr[r] = sum;
    }
    // Pout << "Exit dfCSRMatrix::SpMV(scalar* const __restrict__ ApsiPtr, const scalar* const __restrict__ psiPtr)" << endl << flush;
}

void dfCSRMatrix::SumA(scalar* const __restrict__ sumAPtr) const {
    const scalar* const __restrict__ diagPtr = diag().begin(); 
    #pragma omp parallel for
    for(label r = 0; r < n_; ++r){
        scalar sum = diagPtr[r];
        for(label j = rowPtr_[r]; j < rowPtr_[r + 1]; ++j){
            sum += values_[j];
        }
        sumAPtr[r] = sum;
    }
}

// block Jacobi
void dfCSRMatrix::GaussSeidel(scalar* const __restrict__ psiPtr, scalar* const __restrict__ bPrimePtr) const {
    // Pout << "Enter dfCSRMatrix::GaussSeidel(scalar* const __restrict__ psiPtr, scalar* const __restrict__ bPrimePtr)" << endl << flush;
    const scalar* const __restrict__ diagPtr = diag().begin();

    for(label r = 0; r < n_; ++r){
        scalar sum = bPrimePtr[r];
        for(label j = rowPtr_[r]; j < rowPtr_[r + 1]; ++j){
            sum -= values_[j] * psiPtr[colIdx_[j]];
        }
        psiPtr[r] = sum / diagPtr[r];
    }
    // Pout << "Exit dfCSRMatrix::GaussSeidel(scalar* const __restrict__ psiPtr, scalar* const __restrict__ bPrimePtr)" << endl << flush;
}

void dfCSRMatrix::Jacobi(scalar* const __restrict__ psiPtr, scalar* const __restrict__ bPrimePtr) const {
    // Pout << "Enter dfCSRMatrix::Jacobi(scalar* const __restrict__ psiPtr, scalar* const __restrict__ bPrimePtr)" << endl << flush;
    const scalar* const __restrict__ diagPtr = diag().begin();

    scalar* psiOldPtr = new scalar[n_];
    std::copy(psiPtr, psiPtr + n_, psiOldPtr);
    #pragma omp parallel for
    for(label r = 0; r < n_; ++r){
        scalar sum = bPrimePtr[r];
        for(label j = rowPtr_[r]; j < rowPtr_[r + 1]; ++j){
            sum -= values_[j] * psiOldPtr[colIdx_[j]];
        }
        psiPtr[r] = sum / diagPtr[r];
    }
    delete [] psiOldPtr;
    // Pout << "Exit dfCSRMatrix::Jacobi(scalar* const __restrict__ psiPtr, scalar* const __restrict__ bPrimePtr)" << endl << flush;
}

void dfCSRMatrix::calcDILUReciprocalD(scalar* const __restrict__ rDPtr) const {
    Info << "Not Implemented Error !!! In " << __FILE__ << ":" << __FUNCTION__ << endl << flush;
    MPI_Abort(MPI_COMM_WORLD, -1);
}

void dfCSRMatrix::DILUPrecondition(scalar* const __restrict__ wAPtr, const scalar* const __restrict__ rAPtr, const scalar* const __restrict__ rDPtr) const {
    Info << "Not Implemented Error !!! In " << __FILE__ << ":" << __FUNCTION__ << endl << flush;
    MPI_Abort(MPI_COMM_WORLD, -1);
}

void dfCSRMatrix::DILUPreconditionT(scalar* const __restrict__ wTPtr, const scalar* const __restrict__ rTPtr, const scalar* const __restrict__ rDPtr) const{
    Info << "Not Implemented Error !!! In " << __FILE__ << ":" << __FUNCTION__ << endl << flush;
    MPI_Abort(MPI_COMM_WORLD, -1);
}

void dfCSRMatrix::calcDICReciprocalD(scalar* const __restrict__ rDPtr) const {
    Info << "Not Implemented Error !!! In " << __FILE__ << ":" << __FUNCTION__ << endl << flush;
    MPI_Abort(MPI_COMM_WORLD, -1);
}

void dfCSRMatrix::DICPrecondition(scalar* const __restrict__ wAPtr, const scalar* const __restrict__ rAPtr, const scalar* const __restrict__ rDPtr) const {
    Info << "Not Implemented Error !!! In " << __FILE__ << ":" << __FUNCTION__ << endl << flush;
    MPI_Abort(MPI_COMM_WORLD, -1);
}

} // End namespace Foam
