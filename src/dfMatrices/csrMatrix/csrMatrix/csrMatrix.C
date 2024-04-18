#include "csrMatrix.H"
#include <cassert>
#include <sstream>
#include <cstdio>
#include "PstreamGlobals.H"
#include "Residuals.H"
#include <vector>

namespace Foam{

defineTypeNameAndDebug(csrMatrix, 1);


// csrMatrix::csrMatrix(const Foam::fvMesh& mesh):mesh_(mesh){
//     const labelUList& owner = mesh.owner();
//     const labelUList& neighbour = mesh.neighbour();
//     assert(owner.size() == neighbour.size());
//     this->row_ = mesh.nCells();
//     this->col_ = mesh.nCells();
//     label face_count = neighbour.size();
//     this->off_diag_nnz_ = face_count * 2;

//     this->diag_value_.resize(row_);
//     this->off_diag_rowptr_.resize(row_ + 1);
//     this->off_diag_colidx_.resize(off_diag_nnz_);
//     this->off_diag_value_.resize(off_diag_nnz_);
//     this->face2lower_.resize(face_count);
//     this->face2upper_.resize(face_count);

//     // compute off_diag_count
//     std::vector<label> off_diag_count(row_, 0);
//     std::vector<label> off_diag_current_index(row_ + 1);
//     for(label i = 0; i < face_count; ++i){
//         auto own = owner[i];
//         auto nei = neighbour[i];
//         off_diag_count[own] += 1;
//         off_diag_count[nei] += 1;
//     }

//     off_diag_rowptr_[0] = 0;
//     off_diag_current_index[0] = 0;
//     for(label i = 0; i < row_; ++i){
//         off_diag_rowptr_[i + 1] = off_diag_rowptr_[i] + off_diag_count[i];
//         off_diag_current_index[i + 1] = off_diag_rowptr_[i] + off_diag_count[i];
//     }

//     // compute map from face to lower
//     // (nei, own)
//     for(label i = 0; i < face_count; ++i){
//         label r = neighbour[i];
//         label c = owner[i];
//         label index = off_diag_current_index[r];
//         face2lower_[i] = index;
//         off_diag_colidx_[index] = c;
//         off_diag_current_index[r] += 1;
//     }

//     // (own, nei)
//     for(label i = 0; i < face_count; ++i){
//         label r = owner[i];
//         label c = neighbour[i];
//         label index = off_diag_current_index[r];
//         face2upper_[i] = index;
//         off_diag_colidx_[index] = c;
//         off_diag_current_index[r] += 1;
//     }
// }

// void csrMatrix::init_value_from_lduMatrix(const lduMatrix& ldu){
//     diagonal_ = ldu.diagonal();
//     symmetric_ = ldu.symmetric();
//     asymmetric_ = ldu.asymmetric();

//     const scalarField& ldu_diag_ = ldu.diag();
//     const scalarField& ldu_lower = ldu.lower();
//     const scalarField& ldu_upper_ = ldu.upper();

//     assert(ldu_diag_.size() == diag_value_.size());
//     std::copy(ldu_diag_.begin(),ldu_diag_.end(),diag_value_.begin());

//     assert(ldu_lower.size() == face2lower_.size());
//     assert(ldu_upper_.size() == face2upper_.size());

//     label face_count = mesh().neighbour().size();

//     #pragma omp parallel for
//     for(label i = 0; i < face_count; ++i){
//         off_diag_value_[face2lower_[i]] = ldu_lower[i];
//         off_diag_value_[face2upper_[i]] = ldu_upper_[i];
//     }
// }

csrMatrix::csrMatrix(const lduMatrix& ldu):lduMatrix_(ldu){
    diagonal_ = ldu.diagonal();
    symmetric_ = ldu.symmetric();
    asymmetric_ = ldu.asymmetric();

    const auto& lduDiag = ldu.diag();
    const auto& lduLower = ldu.lower();
    const auto& lduUpper = ldu.upper();
    const auto& lduLowerAddr = ldu.lduAddr().lowerAddr();
    const auto& lduUpperAddr = ldu.lduAddr().upperAddr();

    this->row_ = lduDiag.size();
    this->col_ = lduDiag.size();
    this->off_diag_nnz_ = lduLower.size() + lduUpper.size();

    this->diag_value_.resize(row_);
    this->off_diag_rowptr_.resize(row_ + 1);
    this->off_diag_colidx_.resize(off_diag_nnz_);
    this->off_diag_value_.resize(off_diag_nnz_);

    // fill diag value
    for(label i = 0; i < row_; ++i){
        diag_value_[i] = lduDiag[i];
    }
    // compute off_diag_count
    std::vector<label> off_diag_count(row_, 0);
    std::vector<label> off_diag_current_index(row_ + 1);
    for(label i = 0; i < lduLower.size(); ++i){
        label row = lduUpperAddr[i];
        off_diag_count[row] += 1;
    }
    for(label i = 0; i < lduUpper.size(); ++i){
        label row = lduLowerAddr[i];
        off_diag_count[row] += 1;
    }
    // compute off_diag_rowptr_
    off_diag_rowptr_[0] = 0;
    off_diag_current_index[0] = 0;
    for(label i = 0; i < row_; ++i){
        off_diag_rowptr_[i + 1] = off_diag_rowptr_[i] + off_diag_count[i];
        off_diag_current_index[i + 1] = off_diag_rowptr_[i] + off_diag_count[i];
    }
    assert(off_diag_rowptr_[row_] == off_diag_nnz_);
    // fill non-zero value
    for(label i = 0; i < lduLower.size(); ++i){
        label row = lduUpperAddr[i];
        label index = off_diag_current_index[row];
        off_diag_colidx_[index] = lduLowerAddr[i];
        off_diag_value_[index] = lduLower[i];
        off_diag_current_index[row] += 1;
    }
    for(label i = 0; i < lduUpper.size(); ++i){
        label row = lduLowerAddr[i];
        label index = off_diag_current_index[row];
        off_diag_colidx_[index] = lduUpperAddr[i];
        off_diag_value_[index] = lduUpper[i];
        off_diag_current_index[row] += 1;
    }
}

void csrMatrix::write_pattern(const std::string& filename) const {
    int mpisize, mpirank;
    MPI_Comm_rank(PstreamGlobals::MPI_COMM_FOAM, &mpirank);
    MPI_Comm_size(PstreamGlobals::MPI_COMM_FOAM, &mpisize);
    if(mpirank == 0){
        std::stringstream ss;
        ss << filename << "_" << mpirank << ".mtx";
        FILE* fr = fopen(ss.str().c_str(),"r");
        if(fr != NULL){
            fclose(fr);
            return;
        }
        FILE* fw = fopen(ss.str().c_str(),"w");
        fprintf(fw,"%s\n", "%%MatrixMarket matrix coordinate pattern general");
        fprintf(fw, "%d %d %d\n", row_, col_, off_diag_nnz_);
        for(label r = 0; r < row_; ++r){
            for(label index = off_diag_rowptr_[r]; index < off_diag_rowptr_[r+1]; ++index){
                label c = off_diag_colidx_[index];
                fprintf(fw, "%d %d\n", r + 1, c + 1);
            }
        }
        fclose(fw);
    }
}

void csrMatrix::write(const std::string& filename) const {
    // int mpisize, mpirank;
    // MPI_Comm_rank(PstreamGlobals::MPI_COMM_FOAM, &mpirank);
    // MPI_Comm_size(PstreamGlobals::MPI_COMM_FOAM, &mpisize);
    // if(mpirank == 0){
        int index = 0;
        while(true){
            std::stringstream ss;
            ss << filename << "_matrix_" << row_ << "_" << index << ".mtx";
            FILE* fr = fopen(ss.str().c_str(),"r");
            if(fr != NULL){
                fclose(fr);
            }else{
                break;
            }
            index += 1;
        } 

        std::stringstream ss;
        ss << filename << "_matrix_" << row_ << "_" << index << ".mtx";
        FILE* fw = fopen(ss.str().c_str(),"w");
        
        fprintf(fw,"%s\n", "%%MatrixMarket matrix coordinate real general");
        fprintf(fw, "%d %d %d\n", row_, col_, off_diag_nnz_+row_);
        Info << "csrMatrix::write 3" << endl;
        for(label r = 0; r < row_; ++r){
            for(label index = off_diag_rowptr_[r]; index < off_diag_rowptr_[r+1]; ++index){
                label c = off_diag_colidx_[index];
                scalar v = off_diag_value_[index];
                if(c < r){
                    fprintf(fw, "%d %d %20.16g\n", r + 1, c + 1, v);
                }
            }
            fprintf(fw, "%d %d %20.16g\n", r + 1, r + 1, diag_value_[r]);
            for(label index = off_diag_rowptr_[r]; index < off_diag_rowptr_[r+1]; ++index){
                label c = off_diag_colidx_[index];
                scalar v = off_diag_value_[index];
                if(c > r){
                    fprintf(fw, "%d %d %20.16g\n", r + 1, c + 1, v);
                }
            }
        }
        Info << "csrMatrix::write 4" << endl;
        fclose(fw);
    // }
}

void csrMatrix::write_equation(const std::string& filename,const scalarField& source) const {
    write(filename);

    // int mpisize, mpirank;
    // MPI_Comm_rank(PstreamGlobals::MPI_COMM_FOAM, &mpirank);
    // MPI_Comm_size(PstreamGlobals::MPI_COMM_FOAM, &mpisize);
    // if(mpirank == 0){
        int index = 0;
        while(true){
            std::stringstream ss;
            ss << filename << "_source_" << source.size() << "_" << index << ".dat";
            FILE* fr = fopen(ss.str().c_str(),"r");
            if(fr != NULL){
                fclose(fr);
            }else{
                break;
            }
            index += 1;
        } 
        std::stringstream ss;
        ss << filename << "_source_" << source.size() << "_" << index << ".dat";
        FILE* fw = fopen(ss.str().c_str(),"w");
        fprintf(fw, "%%source vector, the first line is the length of vector, and the following length lines are the value of source vector\n");
        fprintf(fw, "%d \n", source.size());
        forAll(source, i){
            fprintf(fw, "%20.16g\n", source[i]);
        }        
        
    // }
}

void csrMatrix::analyze(){

    label max_col_count = 0;

    for(label r = 0; r < row_; ++r){
        label col_count = off_diag_rowptr_[r+1] - off_diag_rowptr_[r];
        max_col_count = std::max(max_col_count, col_count);
    }

    std::vector<label> col_count_frequency(max_col_count+1,0);

    for(label r = 0; r < row_; ++r){
        label col_count = off_diag_rowptr_[r+1] - off_diag_rowptr_[r];
        col_count_frequency[col_count] += 1;
    }

    Info << "sparse matrix analyze : " << endl;
    Info << "row(col) : " << row_ << endl;
    Info << "nnz : " << off_diag_nnz_ + row_ << endl;
    Info << "sparsity : " << (off_diag_nnz_ + row_) * 1.0 / row_ / row_ * 100  << "%" << endl;
    Info << "max colume count : " << max_col_count << endl;
    Info << "colume frequency : " << endl;
    for(size_t i = 0; i < col_count_frequency.size(); ++i){
        if(col_count_frequency[i] > 0){
            Info << "\t" << i << " : " << col_count_frequency[i] << endl;
        }
    }
    Info << endl;

}


}
