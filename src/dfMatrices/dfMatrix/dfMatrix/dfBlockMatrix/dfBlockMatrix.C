#include "dfBlockMatrix.H"
#include <cassert>
#include <mpi.h>
#include "dfCSRSubMatrix.H"
#include "env.H"
#include <omp.h>

namespace Foam{

void dfBlockMatrix::buildBlocks(const lduMatrix& ldu){
    // Info << "Enter dfBlockMatrix::buildBlocks" << endl;
    std::vector<std::vector<std::tuple<label,label,scalar>>> blocksTmp(rowBlockCount_ * rowBlockCount_);
    const labelList& lduLowerAddr = ldu.lduAddr().lowerAddr();
    const labelList& lduUpperAddr = ldu.lduAddr().upperAddr();
    // lower[i] (lduUpperAddr[i], lduLowerAddr[i]) 
    const scalarList& lduLower = ldu.lower();
    // upper[i] (lduLowerAddr[i], lduUpperAddr[i])
    const scalarList& lduUpper = ldu.upper();

    // lower
    // (lduUpperAddr[i], lduLowerAddr[i])
    for(label i = 0; i < lduLower.size(); ++i){
        label r = lduUpperAddr[i];
        label c = lduLowerAddr[i];
        scalar v = lduLower[i];
        label rbid = std::upper_bound(rowBlockPtr_.begin(), rowBlockPtr_.end(), r) - rowBlockPtr_.begin() - 1;
        label cbid = std::upper_bound(rowBlockPtr_.begin(), rowBlockPtr_.end(), c) - rowBlockPtr_.begin() - 1;
        label rowStart = rowBlockPtr_[rbid];
        label colStart = rowBlockPtr_[cbid];
        blocksTmp[bid2d(rbid, cbid)].push_back({r - rowStart, c - colStart, v});
    }

    // upper
    // (lduLowerAddr[i], lduUpperAddr[i])
    for(label i = 0; i < lduUpper.size(); ++i){
        label r = lduLowerAddr[i];
        label c = lduUpperAddr[i];
        scalar v = lduUpper[i];
        label rbid = std::upper_bound(rowBlockPtr_.begin(), rowBlockPtr_.end(), r) - rowBlockPtr_.begin() - 1;
        label cbid = std::upper_bound(rowBlockPtr_.begin(), rowBlockPtr_.end(), c) - rowBlockPtr_.begin() - 1;
        label rowStart = rowBlockPtr_[rbid];
        label colStart = rowBlockPtr_[cbid];
        blocksTmp[bid2d(rbid, cbid)].push_back({r - rowStart, c - colStart, v});
    }

    // convert each block to CSR
    for(label rbid = 0; rbid < rowBlockCount_; ++rbid){
        for(label cbid = 0; cbid < rowBlockCount_; ++cbid){
            label bid = bid2d(rbid, cbid);
            const auto& block = blocksTmp[bid];
            if(block.size() == 0){
                continue;
            }
            label rowLen = rowBlockPtr_[rbid + 1] - rowBlockPtr_[rbid];
            label colLen = rowBlockPtr_[cbid + 1] - rowBlockPtr_[cbid];
            blocks_[bid] = std::make_unique<dfCSRSubMatrix>(rowLen, colLen, block);
        }
    }
    // Info << "Exit dfBlockMatrix::buildBlocks" << endl;
}

void dfBlockMatrix::buildBlocks(const lduMesh& mesh){
    std::vector<std::vector<std::tuple<label,label,label>>> blocksTmp(rowBlockCount_ * rowBlockCount_);
    const labelList& lduLowerAddr = mesh.lduAddr().lowerAddr();
    const labelList& lduUpperAddr = mesh.lduAddr().upperAddr();
    // lower[i] (lduUpperAddr[i], lduLowerAddr[i]) 
    // const scalarList& lduLower = ldu.lower();
    // upper[i] (lduLowerAddr[i], lduUpperAddr[i])
    // const scalarList& lduUpper = ldu.upper();

    label nFaces = lduUpperAddr.size();

    // lower
    // (lduUpperAddr[i], lduLowerAddr[i])
    for(label i = 0; i < nFaces; ++i){
        label r = lduUpperAddr[i];
        label c = lduLowerAddr[i];
        // scalar v = lduLower[i];
        label rbid = std::upper_bound(rowBlockPtr_.begin(), rowBlockPtr_.end(), r) - rowBlockPtr_.begin() - 1;
        label cbid = std::upper_bound(rowBlockPtr_.begin(), rowBlockPtr_.end(), c) - rowBlockPtr_.begin() - 1;
        label rowStart = rowBlockPtr_[rbid];
        label colStart = rowBlockPtr_[cbid];
        blocksTmp[bid2d(rbid, cbid)].push_back({r - rowStart, c - colStart, i});
    }

    // upper
    // (lduLowerAddr[i], lduUpperAddr[i])
    for(label i = 0; i < nFaces; ++i){
        label r = lduLowerAddr[i];
        label c = lduUpperAddr[i];
        // scalar v = lduUpper[i];
        label rbid = std::upper_bound(rowBlockPtr_.begin(), rowBlockPtr_.end(), r) - rowBlockPtr_.begin() - 1;
        label cbid = std::upper_bound(rowBlockPtr_.begin(), rowBlockPtr_.end(), c) - rowBlockPtr_.begin() - 1;
        label rowStart = rowBlockPtr_[rbid];
        label colStart = rowBlockPtr_[cbid];
        blocksTmp[bid2d(rbid, cbid)].push_back({r - rowStart, c - colStart, i});
    }

    // convert each block to CSR
    for(label rbid = 0; rbid < rowBlockCount_; ++rbid){
        for(label cbid = 0; cbid < rowBlockCount_; ++cbid){
            label bid = bid2d(rbid, cbid);
            const auto& block = blocksTmp[bid];
            if(block.size() == 0){
                continue;
            }
            label rowLen = rowBlockPtr_[rbid + 1] - rowBlockPtr_[rbid];
            label colLen = rowBlockPtr_[cbid + 1] - rowBlockPtr_[cbid];
            blocks_[bid] = std::make_unique<dfCSRSubMatrix>(rowLen, colLen, block);
        }
    }
}

dfBlockMatrix::dfBlockMatrix(const lduMatrix& ldu):dfInnerMatrix(ldu){
    rowBlockCount_ = env::REGION_DECOMPOSE_NBLOCKS;
    rowBlockPtr_.resize(rowBlockCount_ + 1);
    for(label bid = 0; bid < rowBlockCount_; bid++){
        rowBlockPtr_[bid] = n_ * bid / rowBlockCount_;
    }
    rowBlockPtr_[rowBlockCount_] = n_;
    blocks_.resize(rowBlockCount_ * rowBlockCount_);
    buildBlocks(ldu);
}

dfBlockMatrix::dfBlockMatrix(const lduMesh& mesh):dfInnerMatrix(mesh){
    rowBlockCount_ = env::REGION_DECOMPOSE_NBLOCKS;
    rowBlockPtr_.resize(rowBlockCount_ + 1);
    for(label bid = 0; bid < rowBlockCount_; bid++){
        rowBlockPtr_[bid] = n_ * bid / rowBlockCount_;
    }
    rowBlockPtr_[rowBlockCount_] = n_;
    blocks_.resize(rowBlockCount_ * rowBlockCount_);
    buildBlocks(mesh);
}

dfBlockMatrix::dfBlockMatrix(const lduMatrix& ldu, const labelList& rowBlockPtr):dfInnerMatrix(ldu),rowBlockCount_(rowBlockPtr.size()-1),rowBlockPtr_(rowBlockPtr){
    Info << "Building dfBlockMatrix n_ : " << n_ << " rowBlockCount_ : " << rowBlockCount_ << endl;
    blocks_.resize(rowBlockCount_ * rowBlockCount_);
    buildBlocks(ldu);
}

dfBlockMatrix::dfBlockMatrix(const lduMesh& mesh, const labelList& rowBlockPtr):dfInnerMatrix(mesh),rowBlockCount_(rowBlockPtr.size()-1),rowBlockPtr_(rowBlockPtr){
    Info << "Building dfBlockMatrix with mesh n_ : " << n_ << " rowBlockCount_ : " << rowBlockCount_ << endl;
    blocks_.resize(rowBlockCount_ * rowBlockCount_);
    buildBlocks(mesh);
}

// dfBlockMatrix::dfBlockMatrix(const lduMatrix& courseLduMatrix, const labelList& fineRowBlockPtr, const labelList& fineToCoarse):dfInnerMatrix(courseLduMatrix){
//     // this constructor used in Multi-Grid Algorithm
//     rowBlockCount_ = fineRowBlockPtr.size() - 1;
//     rowBlockPtr_.resize(rowBlockCount_ + 1);
//     // rowBlockPtr_[0] = 0;
//     // for(label bid = 0; bid < fineRowBlockPtr.size() - 1; bid++){
//     //     label max_coarseR = -1;
//     //     for(label fineR = fineRowBlockPtr[bid]; fineR < fineRowBlockPtr[bid + 1]; fineR++){
//     //         label coarseR = fineToCoarse[fineR];
//     //         max_coarseR = std::max(max_coarseR, coarseR);
//     //     }
//     //     rowBlockPtr_[bid + 1] = max_coarseR + 1;
//     // }

//     for(label bid = 0; bid < fineRowBlockPtr.size() - 1; bid++){
//         rowBlockPtr_[bid] = n_ * bid / rowBlockCount_;
//     }
//     rowBlockPtr_[rowBlockCount_] = n_;

//     blocks_.resize(rowBlockCount_ * rowBlockCount_);
//     // off_diagonal_nnz_ = 0;
//     if(!courseLduMatrix.hasLower() && !courseLduMatrix.hasUpper()){
//         return;
//     }

//     buildBlocks(courseLduMatrix);
// }

// dfBlockMatrix::dfBlockMatrix(const lduMesh& courseLduMesh, const labelList& fineRowBlockPtr, const labelList& fineToCoarse):dfInnerMatrix(courseLduMesh){
//     // this constructor used in Multi-Grid Algorithm
//     rowBlockCount_ = fineRowBlockPtr.size() - 1;
//     rowBlockPtr_.resize(rowBlockCount_ + 1);
//     for(label bid = 0; bid < fineRowBlockPtr.size() - 1; bid++){
//         rowBlockPtr_[bid] = n_ * bid / rowBlockCount_;
//     }
//     rowBlockPtr_[rowBlockCount_] = n_;
//     blocks_.resize(rowBlockCount_ * rowBlockCount_);
//     buildBlocks(courseLduMesh);
// }

void dfBlockMatrix::valueCopy(const lduMatrix& ldu){
    // Info << "Enter dfBlockMatrix::valueCopy(const lduMatrix& ldu)" << endl << flush;
    dfInnerMatrix::valueCopy(ldu);
    const auto& lower = ldu.lower();
    const auto& upper = ldu.upper();

    // #pragma omp parallel for
    for(label rbid = 0; rbid < rowBlockCount_; ++rbid){
        label rowOffset = rowBlockPtr_[rbid];
        label rowLen = rowBlockPtr_[rbid + 1] - rowBlockPtr_[rbid];
        for(label cbid = 0; cbid < rowBlockCount_; ++cbid){
            label bid = bid2d(rbid, cbid);
            if(blocks_[bid] == nullptr){
                continue;
            }
            if(rbid > cbid){
                blocks_[bid]->valueCopyOffDiagBlock(lower.begin());
            }else if(rbid < cbid){
                blocks_[bid]->valueCopyOffDiagBlock(upper.begin());
            }else{
                blocks_[bid]->valueCopyDiagBlock(lower.begin(), upper.begin());
            }
        }
    }
    // Info << "Exit dfBlockMatrix::valueCopy(const lduMatrix& ldu)" << endl << flush;
}

void dfBlockMatrix::SpMV(scalar* const __restrict__ ApsiPtr, const scalar* const __restrict__ psiPtr) const {
    // Pout << "Enter dfBlockMatrix::SpMV(scalar* const __restrict__ ApsiPtr, const scalar* const __restrict__ psiPtr)" << endl << flush;
    const scalar* const __restrict__ diagPtr = diag().begin();

    #pragma omp parallel for
    for(label rbid = 0; rbid < rowBlockCount_; ++rbid){
        label rowOffset = rowBlockPtr_[rbid];
        label rowLen = rowBlockPtr_[rbid + 1] - rowBlockPtr_[rbid];
        scalar* const __restrict__ ApsiPtr_offset = ApsiPtr + rowOffset;
        const scalar* const __restrict__ diagPtr_offset = diagPtr + rowOffset;
        const scalar* const __restrict__ psiPtr_rowOffset = psiPtr + rowOffset;
        for(label r = 0; r < rowLen; ++r){
            ApsiPtr_offset[r] = diagPtr_offset[r] * psiPtr_rowOffset[r];
        }
        for(label cbid = 0; cbid < rowBlockCount_; ++cbid){
            label bid = bid2d(rbid, cbid);
            if(blocks_[bid] == nullptr){
                continue;
            }
            const dfBlockSubMatrix& csrSubMatrix = *blocks_[bid];
            label colOffset = rowBlockPtr_[cbid];
            csrSubMatrix.SpMV(ApsiPtr_offset, psiPtr + colOffset);
        }
    }
    // Pout << "Exit dfBlockMatrix::SpMV(scalar* const __restrict__ ApsiPtr, const scalar* const __restrict__ psiPtr)" << endl << flush;
}

void dfBlockMatrix::SumA(scalar* const __restrict__ sumAPtr) const {
    const scalar* const __restrict__ diagPtr = diag().begin();

    #pragma omp parallel for
    for(label rbid = 0; rbid < rowBlockCount_; ++rbid){
        label rowOffset = rowBlockPtr_[rbid];
        label rowLen = rowBlockPtr_[rbid + 1] - rowBlockPtr_[rbid];
        scalar* const __restrict__ sumAPtr_offset = sumAPtr + rowOffset;
        const scalar* const __restrict__ diagPtr_offset = diagPtr + rowOffset;
        for(label r = 0; r < rowLen; ++r){
            sumAPtr_offset[r] = diagPtr_offset[r];
        }
        for(label cbid = 0; cbid < rowBlockCount_; ++cbid){
            label bid = bid2d(rbid, cbid);
            if(blocks_[bid] == nullptr){
                continue;
            }
            const dfBlockSubMatrix& csrSubMatrix = *blocks_[bid];
            label colOffset = rowBlockPtr_[cbid];
            csrSubMatrix.SumA(sumAPtr_offset);
        }
    }
}

void dfBlockMatrix::GaussSeidel(scalar* const __restrict__ psiPtr, scalar* const __restrict__ bPrimePtr) const {
    // Pout << "Enter dfBlockMatrix::GaussSeidel(scalar* const __restrict__ psiPtr, scalar* const __restrict__ bPrimePtr)" << endl << flush;
    const scalar* const __restrict__ diagPtr = diag().begin();

    #pragma omp parallel
    {
        int thread_rank = omp_get_thread_num();
        int thread_size = omp_get_num_threads();
        label rb_start = rowBlockCount_ * thread_rank / thread_size;
        label rb_end = rowBlockCount_ * (thread_rank + 1) / thread_size;

        for(label rbid = rb_start; rbid < rb_end; ++rbid){
            label rowOffset = rowBlockPtr_[rbid];
            label rowLen = rowBlockPtr_[rbid+1] - rowBlockPtr_[rbid];
            scalar* const __restrict__ bPrimePtr_offset = bPrimePtr + rowOffset;
            const scalar* const __restrict__ diagPtr_offset = diagPtr + rowOffset;

            // B = b - (L + U) * x
            for(label cbid = 0; cbid < rowBlockCount_; ++cbid){
                label bid = bid2d(rbid, cbid);
                if(rbid == cbid)
                    continue;
                if(blocks_[bid] == nullptr)
                    continue;
                label colOffset = rowBlockPtr_[cbid];
                const dfBlockSubMatrix& offDiagBlock = *blocks_[bid];
                offDiagBlock.BsubApsi(bPrimePtr_offset, psiPtr + colOffset);
            }
        }

        #pragma omp barrier

        for(label rbid = rb_start; rbid < rb_end; ++rbid){
            label rowOffset = rowBlockPtr_[rbid];
            label rowLen = rowBlockPtr_[rbid+1] - rowBlockPtr_[rbid];
            scalar* const __restrict__ bPrimePtr_offset = bPrimePtr + rowOffset;
            const scalar* const __restrict__ diagPtr_offset = diagPtr + rowOffset;

            label diagBlockIndex = bid2d(rbid, rbid);
            scalar* const __restrict__ psiPtr_offset = psiPtr + rowOffset;
            if(blocks_[diagBlockIndex] == nullptr){
                //  x = B / diag
                for(label r = 0; r < rowLen; ++r){
                    psiPtr_offset[r] = bPrimePtr_offset[r] / diagPtr_offset[r];
                }
            }else{
                const dfBlockSubMatrix& diagBlock = *blocks_[diagBlockIndex];
                diagBlock.GaussSeidel(psiPtr_offset, bPrimePtr_offset, diagPtr_offset);
            }
        }
    }
    // for(label rbid = 0; rbid < rowBlockCount_; ++rbid){
    //     label rowOffset = rowBlockPtr_[rbid];
    //     label rowLen = rowBlockPtr_[rbid+1] - rowBlockPtr_[rbid];
    //     scalar* const __restrict__ bPrimePtr_offset = bPrimePtr + rowOffset;
    //     const scalar* const __restrict__ diagPtr_offset = diagPtr + rowOffset;

    //     // B = b - (L + U) * x
    //     for(label cbid = 0; cbid < rowBlockCount_; ++cbid){
    //         label bid = bid2d(rbid, cbid);
    //         if(rbid == cbid)
    //             continue;
    //         if(blocks_[bid] == nullptr)
    //             continue;
    //         label colOffset = rowBlockPtr_[cbid];
    //         const dfBlockSubMatrix& offDiagBlock = *blocks_[bid];
    //         offDiagBlock.BsubApsi(bPrimePtr_offset, psiPtr + colOffset);
    //     }

    //     label diagBlockIndex = bid2d(rbid, rbid);
    //     scalar* const __restrict__ psiPtr_offset = psiPtr + rowOffset;
    //     if(blocks_[diagBlockIndex] == nullptr){
    //         //  x = B / diag
    //         for(label r = 0; r < rowLen; ++r){
    //             psiPtr_offset[r] = bPrimePtr_offset[r] / diagPtr_offset[r];
    //         }
    //     }else{
    //         const dfBlockSubMatrix& diagBlock = *blocks_[diagBlockIndex];
    //         diagBlock.GaussSeidel(psiPtr_offset, bPrimePtr_offset, diagPtr_offset);
    //     }
    // }
    // Pout << "Exit dfBlockMatrix::GaussSeidel(scalar* const __restrict__ psiPtr, scalar* const __restrict__ bPrimePtr)" << endl << flush;
}

void dfBlockMatrix::Jacobi(scalar* const __restrict__ psiPtr, scalar* const __restrict__ bPrimePtr) const {
    // Pout << "Enter dfBlockMatrix::Jacobi(scalar* const __restrict__ psiPtr, scalar* const __restrict__ bPrimePtr)" << endl << flush;
    const scalar* const __restrict__ diagPtr = diag().begin();

    std::unique_ptr<scalar[]> psiOldPtr = std::make_unique<scalar[]>(n_);
    
    #pragma omp parallel for
    for(label r = 0; r < n_; ++r){
        psiOldPtr[r] = psiPtr[r];
    }

    #pragma omp parallel for
    for(label rbid = 0; rbid < rowBlockCount_; ++rbid){
        label rowOffset = rowBlockPtr_[rbid];
        label rowLen = rowBlockPtr_[rbid+1] - rowBlockPtr_[rbid];
        scalar* const __restrict__ bPrimePtr_offset = bPrimePtr + rowOffset;
        const scalar* const __restrict__ diagPtr_offset = diagPtr + rowOffset;
        // b = b - (L + U) * x
        for(label cbid = 0; cbid < rowBlockCount_; ++cbid){
            label bid = bid2d(rbid, cbid);
            if(blocks_[bid] == nullptr){
                continue;
            }
            label colOffset = rowBlockPtr_[cbid];
            const dfBlockSubMatrix& offDiagBlock = *blocks_[bid];
            offDiagBlock.BsubApsi(bPrimePtr_offset, psiOldPtr.get() + colOffset);
        }
        // x = B / diag
        scalar* const __restrict__ psiPtr_offset = psiPtr + rowOffset;
        for(label r = 0; r < rowLen; ++r){
            psiPtr_offset[r] = bPrimePtr_offset[r] / diagPtr_offset[r];
        }
    }

    // Pout << "Exit dfBlockMatrix::Jacobi(scalar* const __restrict__ psiPtr, scalar* const __restrict__ bPrimePtr)" << endl << flush;
}

void dfBlockMatrix::calcDILUReciprocalD(scalar* const __restrict__ rDPtr) const {
    Info << "Not Implemented Error !!! In " << __FILE__ << ":" << __FUNCTION__ << endl << flush;
    MPI_Abort(MPI_COMM_WORLD, -1);
}

void dfBlockMatrix::DILUPrecondition(scalar* const __restrict__ wAPtr, const scalar* const __restrict__ rAPtr, const scalar* const __restrict__ rDPtr) const {
    Info << "Not Implemented Error !!! In " << __FILE__ << ":" << __FUNCTION__ << endl << flush;
    MPI_Abort(MPI_COMM_WORLD, -1);
}

void dfBlockMatrix::DILUPreconditionT(scalar* const __restrict__ wTPtr, const scalar* const __restrict__ rTPtr, const scalar* const __restrict__ rDPtr) const{
    Info << "Not Implemented Error !!! In " << __FILE__ << ":" << __FUNCTION__ << endl << flush;
    MPI_Abort(MPI_COMM_WORLD, -1);
}

void dfBlockMatrix::calcDICReciprocalD(scalar* const __restrict__ rDPtr) const {
    Info << "Not Implemented Error !!! In " << __FILE__ << ":" << __FUNCTION__ << endl << flush;
    MPI_Abort(MPI_COMM_WORLD, -1);
}

void dfBlockMatrix::DICPrecondition(scalar* const __restrict__ wAPtr, const scalar* const __restrict__ rAPtr, const scalar* const __restrict__ rDPtr) const {
    Info << "Not Implemented Error !!! In " << __FILE__ << ":" << __FUNCTION__ << endl << flush;
    MPI_Abort(MPI_COMM_WORLD, -1);
}

} // End namespace Foam
