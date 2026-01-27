#include "dfCSRSubMatrix.H"
#include <cassert>
#include <arm_sve.h>
#include <hbwmalloc.h>
#include <memkind.h>
#include <numa.h>
#include <numaif.h>
#include <sys/mman.h>
#include <stdio.h>
#include <stdlib.h>

#define HBM_ALIGNED_SIZE (1 << 21)
// #define CSR_SUB_USE_HBM

namespace Foam{

dfCSRSubMatrix::dfCSRSubMatrix(label nRows, label nCols, const std::vector<std::tuple<label,label,scalar>>& rcvList):dfBlockSubMatrix(nRows, nCols){
    // count nnz per row
    std::vector<label> nnzPerRow(nRows, 0);
    for(const auto& entry: rcvList){
        label r = std::get<0>(entry);
        nnzPerRow[r] += 1;
    }

    rowPtr_ = std::make_unique<label[]>(nRows + 1);
    std::vector<label> curIndexPerRow(nRows + 1);

    rowPtr_[0] = 0;
    curIndexPerRow[0] = 0;
    for(label i = 0; i < nRows; ++i){
        rowPtr_[i + 1] = rowPtr_[i] + nnzPerRow[i];
        curIndexPerRow[i + 1] = rowPtr_[i + 1];
    }

    label nnz_block = rowPtr_[nRows];
/*
    colIdx_ = std::make_unique<label[]>(nnz_block);
    values_ = std::make_unique<scalar[]>(nnz_block);
*/
//#ifdef CSR_SUB_USE_HBM
    int length = (nnz_block * sizeof(label) + HBM_ALIGNED_SIZE - 1) / HBM_ALIGNED_SIZE * HBM_ALIGNED_SIZE;
    int length2 = (nnz_block * sizeof(scalar) + HBM_ALIGNED_SIZE - 1) / HBM_ALIGNED_SIZE * HBM_ALIGNED_SIZE;
    colIdx_ = (label *)mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    values_ = (scalar *)mmap(NULL, length2, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
//#else
//    colIdx_ = std::make_unique<label[]>(nnz_block);
//    values_ = std::make_unique<scalar[]>(nnz_block);
//#endif
    for(const auto& entry: rcvList){
        label r = std::get<0>(entry);
        label c = std::get<1>(entry);
        scalar v = std::get<2>(entry);
        label rowIdx = r;
        label idx = curIndexPerRow[rowIdx];
        colIdx_[idx] = c;
        values_[idx] = v;
        curIndexPerRow[rowIdx] += 1;
    }

    value_ldu_idx_ = nullptr;
}

dfCSRSubMatrix::dfCSRSubMatrix(label nRows, label nCols, const std::vector<std::tuple<label,label,label>>& rciList):dfBlockSubMatrix(nRows, nCols){
    // count nnz per row
    std::vector<label> nnzPerRow(nRows, 0);
    for(const auto& entry: rciList){
        label r = std::get<0>(entry);
        nnzPerRow[r] += 1;
    }

    rowPtr_ = std::make_unique<label[]>(nRows + 1);
    std::vector<label> curIndexPerRow(nRows + 1);

    rowPtr_[0] = 0;
    curIndexPerRow[0] = 0;
    for(label i = 0; i < nRows; ++i){
        rowPtr_[i + 1] = rowPtr_[i] + nnzPerRow[i];
        curIndexPerRow[i + 1] = rowPtr_[i + 1];
    }

    label nnz_block = rowPtr_[nRows];
/*
        if(element_num_ > 0) {
            int length = (element_num_ * sizeof(DataType) + HBM_ALIGNED_SIZE - 1) / HBM_ALIGNED_SIZE * HBM_ALIGNED_SIZE;
            void* mapAddress = NULL;
            mapAddress = mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
            if (mapAddress == NULL) {
                printf("mmap failed ,length = %d \n", length);
                exit(-1);
            };
            data_ = (DataType*)mapAddress;
        } else {
            data_ = nullptr;
        }
        printf("using mmap allocating memory\n");
*/
//#ifdef CSR_SUB_USE_HBM
    
    int length = (nnz_block * sizeof(label) + HBM_ALIGNED_SIZE - 1) / HBM_ALIGNED_SIZE * HBM_ALIGNED_SIZE;
    int length2 = (nnz_block * sizeof(scalar) + HBM_ALIGNED_SIZE - 1) / HBM_ALIGNED_SIZE * HBM_ALIGNED_SIZE;
    void* mapAddress = NULL;
    mapAddress = mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (mapAddress == NULL) printf("mmap failed ,colIdx_ NULL, length = %d \n", length);
    colIdx_ = (label *)mapAddress;
    mapAddress = NULL;
    mapAddress = mmap(NULL, length2, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (mapAddress == NULL) printf("mmap failed ,values_ NULL, length2 = %d \n", length2);
    values_ = (scalar *)mapAddress;
    mapAddress = NULL;
    mapAddress = mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (mapAddress == NULL) printf("mmap failed ,value_ldu_idx_ NULL, length = %d \n", length);
    value_ldu_idx_ = (label *)mapAddress;
/*
    colIdx_ = (label *)mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (colIdx_ == NULL) printf("mmap failed ,colIdx_ NULL, length = %d \n", length);
    
    values_ = (scalar *)mmap(NULL, length2, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (values_ == NULL) printf("mmap failed ,values_ NULL, length2 = %d \n", length2);
    
    value_ldu_idx_ = (label *)mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (value_ldu_idx_ == NULL) printf("mmap failed ,value_ldu_idx_ NULL, length = %d \n", length);
    
    printf("using mmap allocating CSR memory 1338\n");
//#else
//    colIdx_ = std::make_unique<label[]>(nnz_block);
//    values_ = std::make_unique<scalar[]>(nnz_block);
//    value_ldu_idx_ = std::make_unique<label[]>(nnz_block);
//#endif

*/
    for(const auto& entry: rciList){
        label r = std::get<0>(entry);
        label c = std::get<1>(entry);
        label faceIndex = std::get<2>(entry);
        label rowIdx = r;
        label idx = curIndexPerRow[rowIdx];
        colIdx_[idx] = c;
        value_ldu_idx_[idx] = faceIndex;
        curIndexPerRow[rowIdx] += 1;
    }
}

void dfCSRSubMatrix::valueCopyOffDiagBlock(const scalar* const __restrict__ lduValuePt){
    // Info << "Enter dfCSRSubMatrix::valueCopyOffDiagBlock(const scalar* const __restrict__ lduValuePt)" << endl << flush;
    const label* const __restrict__ rowPtr = rowPtr_.get();
    const label* const __restrict__ value_ldu_idx_ptr_= value_ldu_idx_;
    scalar* const __restrict__ valuePtr = values_;

    for(label r = 0; r < nRows_; ++r){
        for(label idx = rowPtr[r]; idx < rowPtr[r+1]; ++idx){
            valuePtr[idx] = lduValuePt[value_ldu_idx_ptr_[idx]];
        }
    }
    // Info << "Exit dfCSRSubMatrix::valueCopyOffDiagBlock(const scalar* const __restrict__ lduValuePt)" << endl << flush;
}

void dfCSRSubMatrix::valueCopyDiagBlock(const scalar* const __restrict__ lower, const scalar* const __restrict__ upper){
    // Info << "Enter dfCSRSubMatrix::valueCopyDiagBlock(const scalar* const __restrict__ lower, const scalar* const __restrict__ upper)" << endl << flush;
    const label* const __restrict__ rowPtr = rowPtr_.get();
    const label* const __restrict__ colIdxPtr = colIdx_;
    const label* const __restrict__ value_ldu_idx_ptr_= value_ldu_idx_;
    scalar* const __restrict__ valuePtr = values_;

    for(label r = 0; r < nRows_; ++r){
        for(label idx = rowPtr[r]; idx < rowPtr[r+1]; ++idx){
            label c = colIdxPtr[idx];
            if(r > c){
                valuePtr[idx] = lower[value_ldu_idx_ptr_[idx]];
            }else if(r < c){
                valuePtr[idx] = upper[value_ldu_idx_ptr_[idx]];
            }else{
                assert(false);
            }
        }
    }
    // Info << "Exit dfCSRSubMatrix::valueCopyDiagBlock(const scalar* const __restrict__ lower, const scalar* const __restrict__ upper)" << endl << flush;
}

/*
void dfCSRSubMatrix::SpMV(scalar* const __restrict__ ApsiPtr_offset, const scalar* const __restrict__ psiPtr_offset) const {
    const label* const __restrict__ rowPtr = rowPtr_.get();
    const label* const __restrict__ colIdxPtr = colIdx_.get();
    const scalar* const __restrict__ valuePtr = values_.get();
    
    //Info << "Enter dfCSRSubMatrix::SpMV(const scalar* const __restrict__ lower, const scalar* const __restrict__ upper)" << endl << flush;
    for(label r = 0; r < nRows_; ++r){
        scalar sum = 0.;

	//Info << "r = " << r << ", low = " << rowPtr[r] << ", high = " << rowPtr[r+1] << endl << flush;
        //#pragma clang loop vectorize(enable)
        for(label idx = rowPtr[r]; idx < rowPtr[r+1]; ++idx){
	    //__builtin_prefetch(&colIdxPtr[idx + 4], 0, 3);
	    //__builtin_prefetch(&psiPtr_offset[colIdxPtr[idx + 4]], 0, 3);
            sum += valuePtr[idx] * psiPtr_offset[colIdxPtr[idx]];
        }
        ApsiPtr_offset[r] += sum;
    }
}
*/

inline static scalar spmv_gather_kernel(scalar sum, label low, label high,
    const scalar *rhs_data, const scalar *val_data, const label *col_ptr_data) {
  svbool_t pg = svwhilelt_b64_s64(low, high);
  
  svfloat64_t x_slice = svld1_f64(pg, val_data + low);
  
  svint64_t indices = svld1_s64(pg, col_ptr_data + low);
  svfloat64_t y_slice = svld1_gather_s64index_f64(pg, rhs_data, indices);
  
  svfloat64_t z_slice = svmul_f64_x(pg, x_slice, y_slice);

  return sum + svaddv_f64(pg, z_slice);
}

void dfCSRSubMatrix::SpMV(scalar* const __restrict__ ApsiPtr_offset, const scalar* const __restrict__ psiPtr_offset) const {
    const label* const __restrict__ rowPtr = rowPtr_.get();
    const label* const __restrict__ colIdxPtr = colIdx_;
    const scalar* const __restrict__ valuePtr = values_;
    
    // Info << "Enter dfCSRSubMatrix::SpMV(const scalar* const __restrict__ lower, const scalar* const __restrict__ upper)" << endl << flush;
    for(label r = 0; r < nRows_; ++r){
        scalar sum = 0.;

	label low = rowPtr[r], high = rowPtr[r+1];
	if (low + 8 < high) {
	    Info << "In SpMV: " <<  r << ' ' << low << ' ' << high << '\n';
	}
	assert(low + 8 >= high);
	sum = spmv_gather_kernel(sum, low, high, psiPtr_offset, valuePtr, colIdxPtr);

	ApsiPtr_offset[r] += sum;
    }
}


void dfCSRSubMatrix::SumA(scalar* const __restrict__ sumAPtr_offset) const {
    const label* const __restrict__ rowPtr = rowPtr_.get();
    const scalar* const __restrict__ valuePtr = values_;
    Info << "Enter dfCSRSubMatrix::SumA(scalar* const __restrict__ sumAPtr_offset)" << endl << flush;
    for(label r = 0; r < nRows_; ++r){
        scalar sum = 0.;
        for(label idx = rowPtr[r]; idx < rowPtr[r+1]; ++idx){
            sum += valuePtr[idx];
        }
        sumAPtr_offset[r] += sum;
    }
}


void dfCSRSubMatrix::BsubApsi(scalar* const __restrict__ BPtr_offset, const scalar* const __restrict__ psiPtr_offset) const {
    // Info << "Enter dfCSRSubMatrix::BsubApsi(scalar* const __restrict__ BPtr_offset, const scalar* const __restrict__ psiPtr_offset)" << endl << flush; 
    const label* const __restrict__ rowPtr = rowPtr_.get();
    const label* const __restrict__ colIdxPtr = colIdx_;
    const scalar* const __restrict__ valuePtr = values_;
    for(label r = 0; r < nRows_; ++r){
        scalar sum = 0.0;
        for(label idx = rowPtr[r]; idx < rowPtr[r+1]; ++idx){
            sum += valuePtr[idx] * psiPtr_offset[colIdxPtr[idx]];
        }
        BPtr_offset[r] -= sum;
    }
    // Info << "Exit dfCSRSubMatrix::BsubApsi(scalar* const __restrict__ BPtr_offset, const scalar* const __restrict__ psiPtr_offset)" << endl << flush; 
}

void dfCSRSubMatrix::GaussSeidel(scalar* const __restrict__ psiPtr_offset, const scalar* const __restrict__ BPtr_offset, const scalar* const __restrict__ diagPtr_offset) const {
    // Info << "Enter dfCSRSubMatrix::GaussSeidel(scalar* const __restrict__ psiPtr_offset, const scalar* const __restrict__ BPtr_offset, const scalar* const __restrict__ diagPtr_offset)" << endl << flush;
    const label* const __restrict__ rowPtr = rowPtr_.get();
    const label* const __restrict__ colIdxPtr = colIdx_;
    const scalar* const __restrict__ valuePtr = values_;
    for(label r = 0; r < nRows_; ++r){
        scalar sum = 0.0;
        for(label idx = rowPtr[r]; idx < rowPtr[r+1]; ++idx){
            sum += valuePtr[idx] * psiPtr_offset[colIdxPtr[idx]];
        }
        psiPtr_offset[r] = (BPtr_offset[r] - sum) / diagPtr_offset[r];
    }
    // Info << "Exit dfCSRSubMatrix::GaussSeidel(scalar* const __restrict__ psiPtr_offset, const scalar* const __restrict__ BPtr_offset, const scalar* const __restrict__ diagPtr_offset)" << endl << flush;
}

// void dfCSRSubMatrix::Jacobi(scalar* const __restrict__ psiPtr_offset, const scalar* const __restrict__ psiOldPtr_offset, const scalar* const __restrict__ BPtr_offset, const scalar* const __restrict__ diagPtr_offset) const {
//     Info << "Enter dfCSRSubMatrix::Jacobi(scalar* const __restrict__ psiPtr_offset, const scalar* const __restrict__ psiOldPtr_offset, const scalar* const __restrict__ BPtr_offset, const scalar* const __restrict__ diagPtr_offset)" << endl << flush;
//     const label* const __restrict__ rowPtr = rowPtr_.get();
//     const label* const __restrict__ colIdxPtr = colIdx_.get();
//     const scalar* const __restrict__ valuePtr = values_.get();
//     for(label r = 0; r < nRows_; ++r){
//         scalar sum = 0.0;
//         for(label idx = rowPtr[r]; idx < rowPtr[r+1]; ++idx){
//             sum += valuePtr[idx] * psiOldPtr_offset[colIdx_[idx]];
//         }
//         psiPtr_offset[r] = (BPtr_offset[r] - sum) / diagPtr_offset[r];
//     }
//     Info << "Exit dfCSRSubMatrix::Jacobi(scalar* const __restrict__ psiPtr_offset, const scalar* const __restrict__ psiOldPtr_offset, const scalar* const __restrict__ BPtr_offset, const scalar* const __restrict__ diagPtr_offset)" << endl << flush;
// }

}
