#include "BCSRPattern.H"
#include <algorithm>
#include <cassert>
#include <mpi.h>
#include <cstdio>

namespace Foam{

BlockPattern::BlockPattern(const csrPattern csr, const labelList& regionPtr){

    n_ = csr.n();
    rowBlockCount_ = regionPtr.size() - 1; 
    rowBlockPtr_ = regionPtr;

    blocks_.resize(rowBlockCount_ * rowBlockCount_);

    const auto& csrRowPtr = csr.rowptr();
    const auto& csrColIdx = csr.colidx();

    for(label r = 0; r < csr.n(); ++r){
        label rbid = std::upper_bound(rowBlockPtr_.begin(), rowBlockPtr_.end(), r) - rowBlockPtr_.begin() - 1;
        assert(rbid < rowBlockCount_);
        assert(rbid >= 0);
        for(label idx = csrRowPtr[r]; idx < csrRowPtr[r+1]; ++idx){
            label c = csrColIdx[idx];
            label cbid = std::upper_bound(rowBlockPtr_.begin(), rowBlockPtr_.end(), c) - rowBlockPtr_.begin() - 1;
            assert(cbid < rowBlockCount_);
            assert(cbid >= 0);
            blocks_[bid2d(rbid, cbid)].push_back({r,c});
        }
    }
    show();
}

void BlockPattern::show() const {
    int mpirank, mpisize;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpisize);

    Info << "BlockPattern Info : " << endl;
    Info << "rowBlockCount_ : " << rowBlockCount_ << endl;
    Info << "rowBlockPtr_ : " << rowBlockPtr_ << endl;
    if(mpirank == 0){
        for(label rbid = 0; rbid < rowBlockCount_; ++rbid){
            for(label cbid = 0; cbid < rowBlockCount_; ++cbid){
                // Info << setw(12) << setfill(' ') << blocks_[bid2d(rbid, cbid)].size() << "\t";
                printf("%12d\t", blocks_[bid2d(rbid, cbid)].size());
            }
            printf("\n");
            fflush(stdout);
        }
    }

    label off_diagonal_block_count = 0;
    label total_nnz = 0;
    label off_diagonal_block_nnz = 0;
    for(label rbid = 0; rbid < rowBlockCount_; ++rbid){
        for(label cbid = 0; cbid < rowBlockCount_; ++cbid){
            total_nnz += blocks_[bid2d(rbid, cbid)].size();
            if(rbid != cbid){
                if (blocks_[bid2d(rbid, cbid)].size() > 0){
                    off_diagonal_block_count++;
                }
                off_diagonal_block_nnz += blocks_[bid2d(rbid, cbid)].size();
            }
        }
    }

    Info << "off_diagonal_block_count : " << off_diagonal_block_count << endl;
    Info << "off_diagonal_block_nnz : " << off_diagonal_block_nnz << endl;
    Info << "total_nnz : " << total_nnz << endl;
    Info << "off_diagonal_block_nnz / total_nnz : " << (double)off_diagonal_block_nnz / total_nnz * 100 << "%" << endl;
}

}
