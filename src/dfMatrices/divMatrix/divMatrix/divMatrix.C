#include "divMatrix.H"
#include <cassert>
#include <sstream>
#include <cstdio>
#include "PstreamGlobals.H"
#include "Residuals.H"
#include <vector>
#include "env.H"

namespace Foam{

defineTypeNameAndDebug(divMatrix, 0);

divMatrix::divMatrix(const lduMatrix& ldu):lduMatrix_(&ldu),row_block_bit_(row_block_bit), row_block_size_(1 << row_block_bit_){
    diagonal_ = ldu.diagonal();
    symmetric_ = ldu.symmetric();
    asymmetric_ = ldu.asymmetric();

    const auto& lduDiag = ldu.diag();
    const auto& lduLower = ldu.lower();
    const auto& lduUpper = ldu.upper();
    const auto& lduLowerAddr = ldu.lduAddr().lowerAddr();
    const auto& lduUpperAddr = ldu.lduAddr().upperAddr();

    row_ = ldu.lduAddr().size();
    block_count_ = DIV_BLOCK_COUNT;
    block_tail_ = DIV_BLOCK_TAIL;
    assert(block_tail_ == 0);
    // fill diag value
    diag_value_.resize(row_);
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for(label i = 0; i < row_; ++i){
        diag_value_[i] = lduDiag[i];
    }
    std::vector<label> distance_map(2 * row_ - 1, 0);
    distance_count_ = 0;
    for(label i = 0; i < lduLower.size(); ++i){
        label r = lduUpperAddr[i];
        label c = lduLowerAddr[i];
        label distance = c - r;
        if(distance_map[distance + row_ - 1] == 0){
            distance_count_ += 1;
            distance_map[distance + row_ - 1] = 1;
        }
    }
    for(label i = 0; i < lduUpper.size(); ++i){
        label r = lduLowerAddr[i];
        label c = lduUpperAddr[i];
        label distance = c - r;
        if(distance_map[distance + row_ - 1] == 0){
            distance_count_ += 1;
            distance_map[distance + row_ - 1] = 1;
        }
    }
    distance_list_.resize(distance_count_);
    label distance_index = 0;
    for(size_t i = 0 ; i < distance_map.size(); ++i){
        if(distance_map[i] > 0){
            distance_list_[distance_index] = i - row_ + 1;
            distance_map[i] = distance_index;
            distance_index += 1;
        }else{
            distance_map[i] = -1;
        }
    }
    max_distance_ = 0;
    min_distance_ = 0;
    for(size_t i = 0; i < distance_list_.size(); ++i){
        max_distance_ = std::max(max_distance_, distance_list_[i]);
        min_distance_ = std::min(min_distance_, distance_list_[i]);
    }
    head_block_count_ = (- min_distance_ + row_block_size_ - 1) / row_block_size_;
    tail_block_count_ = (max_distance_ + row_block_size_ - 1) / row_block_size_;
    // fill off_diag_value
    off_diag_value_.resize(distance_count_ * row_);
    for(label i = 0 ; i < off_diag_value_.size(); ++i){
        off_diag_value_[i] = 0.;
    }
    for(label i = 0; i < lduLower.size(); ++i){
        label row = lduUpperAddr[i];
        label col = lduLowerAddr[i];
        label distance = col - row;
        label divcol = distance_map[distance + row_ - 1];
        assert(divcol != -1);
        assert(divcol < distance_count_);
        label index = DIV_INDEX(row, divcol);
        off_diag_value_[index] = lduLower[i];
    }
    for(label i = 0; i < lduUpper.size(); ++i){
        label row = lduLowerAddr[i];
        label col = lduUpperAddr[i];
        label distance = col - row;
        label divcol = distance_map[distance + row_ - 1];
        assert(divcol != -1);
        assert(divcol < distance_count_);
        label index = DIV_INDEX(row, divcol);
        off_diag_value_[index] = lduUpper[i];
    }
}

divMatrix::divMatrix(const lduMesh& mesh):row_block_bit_(row_block_bit), row_block_size_(1 << row_block_bit_){
    const labelUList& lduLowerAddr = mesh.lduAddr().lowerAddr();
    const labelUList& lduUpperAddr = mesh.lduAddr().upperAddr();
    row_ = mesh.lduAddr().size();
    label face_count = lduUpperAddr.size();

    block_count_ = DIV_BLOCK_COUNT;
    block_tail_ = DIV_BLOCK_TAIL;
    assert(block_tail_ == 0);

    // fill diag value
    diag_value_.resize(row_);

    std::vector<label> distance_map(2 * row_ - 1, 0);
    distance_count_ = 0;
    for(label i = 0; i < face_count; ++i){
        label r = lduUpperAddr[i];
        label c = lduLowerAddr[i];
        label distance0 = c - r;
        label distance1 = r - c;
        if(distance_map[distance0 + row_ - 1] == 0){
            distance_count_ += 1;
            distance_map[distance0 + row_ - 1] = 1;
        }
        if(distance_map[distance1 + row_ - 1] == 0){
            distance_count_ += 1;
            distance_map[distance1 + row_ - 1] = 1;
        }
    }

    distance_list_.resize(distance_count_);
    label distance_index = 0;
    for(size_t i = 0 ; i < distance_map.size(); ++i){
        if(distance_map[i] > 0){
            distance_list_[distance_index] = i - row_ + 1;
            distance_map[i] = distance_index;
            distance_index += 1;
        }else{
            distance_map[i] = -1;
        }
    }

    max_distance_ = 0;
    min_distance_ = 0;
    for(size_t i = 0; i < distance_list_.size(); ++i){
        max_distance_ = std::max(max_distance_, distance_list_[i]);
        min_distance_ = std::min(min_distance_, distance_list_[i]);
    }
    head_block_count_ = (- min_distance_ + row_block_size_ - 1) / row_block_size_;
    tail_block_count_ = (max_distance_ + row_block_size_ - 1) / row_block_size_;
    
    // alloc off_diag_value
    off_diag_value_.resize(distance_count_ * row_);

    face2lower_.resize(face_count);
    face2upper_.resize(face_count);

    for(label i = 0; i < face_count; ++i){
        label row = lduUpperAddr[i];
        label col = lduLowerAddr[i];
        label distance = col - row;
        label divcol = distance_map[distance + row_ - 1];
        assert(divcol != -1);
        assert(divcol < distance_count_);
        label index = DIV_INDEX(row, divcol);
        // off_diag_value_[index] = lduLower[i];
        face2lower_[i] = index;
    }
    for(label i = 0; i < face_count; ++i){
        label row = lduLowerAddr[i];
        label col = lduUpperAddr[i];
        label distance = col - row;
        label divcol = distance_map[distance + row_ - 1];
        assert(divcol != -1);
        assert(divcol < distance_count_);
        label index = DIV_INDEX(row, divcol);
        // off_diag_value_[index] = lduUpper[i];
        face2upper_[i] = index;
    }

    // init comm info
    scalarSendBufList_ = new scalar*[nBoundaryPatches];
    scalarRecvBufList_ = new scalar*[nBoundaryPatches];
    scalarSendBufListGAMG_.resize(nBoundaryPatches);
    scalarRecvBufListGAMG_.resize(nBoundaryPatches);
    for (label patchi = 0; patchi < nBoundaryPatches; patchi++) {
        if (nInterfacesGroup[patchi]) {
            scalarSendBufList_[patchi] = new scalar[nInterfacesGroup[patchi]];
            scalarRecvBufList_[patchi] = new scalar[nInterfacesGroup[patchi]];
        }
    }
    
    Info << "divMatrix info : ----------------------------------" << endl;
    Info << "row_ : " << row_ << endl;
    Info << "block_count_ : " << block_count_ << endl;
    Info << "distance_count_ : " << distance_count_ << endl;
    Info << "distance_list_ : " << endl << "\t";
    for(label i = 0; i < distance_count_; ++i){
        Info << distance_list_[i] << ", ";
    }
    Info << endl;
    Info << "---------------------------------------------------" << endl;
}

void divMatrix::copy_value_from_fvMatrix(const lduMatrix& lduMatrix){
    flops_ = 0.;

    lduMatrix_ = &lduMatrix;

    diagonal_ = lduMatrix.diagonal();
    symmetric_ = lduMatrix.symmetric();
    asymmetric_ = lduMatrix.asymmetric();

    const auto& lduDiag = lduMatrix.diag();
    const auto& lduLower = lduMatrix.lower();
    const auto& lduUpper = lduMatrix.upper();

    assert(row_ == lduMatrix.diag().size());

    assert(face2lower_.size() == lduLower.size());
    assert(face2upper_.size() == lduUpper.size());

    label face_count = lduLower.size();

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for(label i = 0; i < row_; ++i){
        diag_value_[i] = lduDiag[i];
    }

    memset(off_diag_value_.data(), '\0', off_diag_value_.size() * sizeof(scalar));

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for(label i = 0; i < face_count; ++i){
        off_diag_value_[face2lower_[i]] = lduLower[i];
        off_diag_value_[face2upper_[i]] = lduUpper[i];
    }
}

void divMatrix::analyze() const {
}

void divMatrix::check() const {
    // const auto& lduDiag = ldu().diag();
    const auto& lduLower = ldu().lower();
    const auto& lduUpper = ldu().upper();
    const auto& lduLowerAddr = ldu().lduAddr().lowerAddr();
    const auto& lduUpperAddr = ldu().lduAddr().upperAddr();
    const scalar* const __restrict__ diagPtr = diag_value_.begin();
    const scalar* const __restrict__ off_diag_value_Ptr = off_diag_value_.begin();
    #pragma omp parallel for
    for(label bi = 0; bi < block_count_; ++bi){
        label rbs = DIV_BLOCK_START(bi);
        label rbe = DIV_BLOCK_END(rbs);
        label rbl = DIV_BLOCK_LEN(rbs,rbe);
        label index_block_start = DIV_INDEX_BLOCK_START(rbs);
        for(label divcol = 0; divcol < distance_count_; ++divcol){
            label index_divcol_start = index_block_start + DIV_COL_OFFSET(divcol);
            label distance = distance_list_[divcol];
            if(distance + rbs >= 0 && distance + rbe <= row_){
                for(label br = 0; br < rbl; ++br){
                    label row = rbs + br;
                    label col = distance + rbs + br;
                    scalar value = off_diag_value_Ptr[index_divcol_start + br];
                    if(row > col){ // lower part
                        bool found = false;
                        for(label i = 0; i < lduLower.size(); ++i){
                            label rr = lduUpperAddr[i];
                            label cc = lduLowerAddr[i];
                            scalar vv = lduLower[i];
                            if(rr == row && cc == col){
                                found = true;
                                assert(vv == value);
                            }
                        }
                        if(found == false){
                            assert(false);
                        }
                    }else if(col < row){ // upper part
                        bool found = false;
                        for(label i = 0; i < lduUpper.size(); ++i){
                            label rr = lduLowerAddr[i];
                            label cc = lduUpperAddr[i];
                            scalar vv = lduUpper[i];
                            if(rr == row && cc == col){
                                found = true;
                                assert(vv == value);
                            }
                        }
                        if(found == false){
                            assert(false);
                        }
                    }else{
                        assert(false);
                    }
                }
            }else{
                for(label br = 0; br < rbl; ++br){
                    label col = distance + rbs + br;
                    col = std::max(col, static_cast<label>(0));
                    col = std::min(col, row_ - 1);
                    if(col < 0 || col > (row_ - 1)){
                        assert(off_diag_value_Ptr[index_divcol_start + br] == 0);
                    }else{
                        label row = rbs + br;
                        scalar value = off_diag_value_Ptr[index_divcol_start + br];
                        if(row > col){ // lower part
                            bool found = false;
                            for(label i = 0; i < lduLower.size(); ++i){
                                label rr = lduUpperAddr[i];
                                label cc = lduLowerAddr[i];
                                scalar vv = lduLower[i];
                                if(rr == row && cc == col){
                                    found = true;
                                    assert(vv == value);
                                }
                            }
                            if(found == false){
                                assert(false);
                            }
                        }else if(col < row){ // upper part
                            bool found = false;
                            for(label i = 0; i < lduUpper.size(); ++i){
                                label rr = lduLowerAddr[i];
                                label cc = lduUpperAddr[i];
                                scalar vv = lduUpper[i];
                                if(rr == row && cc == col){
                                    found = true;
                                    assert(vv == value);
                                }
                            }
                            if(found == false){
                                assert(false);
                            }
                        }else{
                            assert(false);
                        }
                    }
                }       
            }
        }
    }
}


}
