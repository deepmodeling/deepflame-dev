#include <vector>
#include <cassert>
#include <string>
#include <PstreamGlobals.H>
#include "csrPattern.H"
#include <tuple>

namespace Foam{

csrPattern::csrPattern(const fvMesh& mesh){
    const labelUList& owner = mesh.owner();
    const labelUList& neighbour = mesh.neighbour();
    assert(owner.size() == neighbour.size());
    this->n_ = mesh.nCells();
    label face_count = neighbour.size();
    this->nnz_ = face_count * 2 + n_;

    this->rowptr_.resize(n_ + 1);
    this->colidx_.resize(nnz_);

    // compute row_count
    std::vector<label> row_count(n_, 1);
    std::vector<label> current_index(n_ + 1);
    for(label i = 0; i < face_count; ++i){
        auto own = owner[i];
        auto nei = neighbour[i];
        row_count[own] += 1;
        row_count[nei] += 1;
    }

    rowptr_[0] = 0;
    current_index[0] = 0;
    for(label i = 0; i < n_; ++i){
        rowptr_[i + 1] = rowptr_[i] + row_count[i];
        current_index[i + 1] = rowptr_[i] + row_count[i];
    }

    // lower
    // (nei, own)
    for(label i = 0; i < face_count; ++i){
        label r = neighbour[i];
        label c = owner[i];
        label index = current_index[r];
        // face2lower_[i] = index;
        colidx_[index] = c;
        current_index[r] += 1;
    }

    // diag
    for(label rc = 0; rc < n_; ++rc){
        label index = current_index[rc];
        colidx_[index] = rc;
        current_index[rc] += 1;
    }

    // upper
    // (own, nei)
    for(label i = 0; i < face_count; ++i){
        label r = owner[i];
        label c = neighbour[i];
        label index = current_index[r];
        // face2upper_[i] = index;
        colidx_[index] = c;
        current_index[r] += 1;
    }
}

csrPattern csrPattern::blocking(label block_size){
    csrPattern blocked_matrix;
    assert(n_ % block_size == 0);

    label block_count = n_ / block_size;

    blocked_matrix.n_ = block_count;

    std::vector<label> mark(block_count, -1);
    std::vector<label> row_count(block_count, 0);

    // row count
    for(label br = 0; br < block_count; ++br){
        label bs = br * block_size;
        label be = (br + 1) * block_size;
        for(label r = bs; r < be; ++r){
            for(label index = rowptr_[r]; index < rowptr_[r+1]; ++index){
                label c = colidx_[index];
                label bc = c / block_size;
                mark[bc] = br;
            }
        }
        for(label bc = 0; bc < block_count; ++bc){
            if(mark[bc] == br){
                row_count[br] += 1;
            }
        }
    }

    std::vector<label> cur_index(block_count);
    blocked_matrix.rowptr_.resize(block_count + 1);

    // rowptr nnz
    blocked_matrix.rowptr_[0] = 0;
    for(label br = 0; br < block_count; ++br){
        blocked_matrix.rowptr_[br + 1] = blocked_matrix.rowptr_[br] + row_count[br];
    }
    blocked_matrix.nnz_ = blocked_matrix.rowptr_[block_count];
    blocked_matrix.colidx_.resize(blocked_matrix.nnz_);

    for(label br = 0; br < block_count; ++br){
        cur_index[br] = blocked_matrix.rowptr_[br];
    }

    // reset mark
    for(label br = 0; br < block_count; ++br){
        mark[br] = -1;
    }

    // fill colidx
    for(label br = 0; br < block_count; ++br){
        label bs = br * block_size;
        label be = (br + 1) * block_size;
        for(label r = bs; r < be; ++r){
            for(label index = rowptr_[r]; index < rowptr_[r+1]; ++index){
                label c = colidx_[index];
                label bc = c / block_size;
                if(mark[bc] != br){
                    label bindex = cur_index[br];
                    blocked_matrix.colidx_[bindex] = bc;
                    cur_index[br] += 1;
                    mark[bc] = br;
                }
            }
        }
    }

    // check
    for(label br = 0; br < block_count; ++br){
        assert(cur_index[br] == blocked_matrix.rowptr_[br + 1]);
    }

    return blocked_matrix;
}

csrPattern csrPattern::lower_part(){
    csrPattern ret;
    ret.n_ = n_;

    // compute row_count
    std::vector<label> row_count(n_, 0);
    for(label r = 0; r < n_; ++r){
        for(label index = rowptr_[r]; index < rowptr_[r + 1]; ++index){
            label c = colidx_[index];
            if(c < r){
                row_count[r] += 1;
            }
        }
    }

    // rowptr nnz
    ret.rowptr_.resize(n_ + 1);
    ret.rowptr_[0] = 0;
    for(label r = 0; r < n_; ++r){
        ret.rowptr_[r + 1] = ret.rowptr_[r] + row_count[r];
    }
    ret.nnz_ = ret.rowptr_[n_];
    ret.colidx_.resize(ret.nnz_);

    std::vector<label> cur_index(n_);
    for(label r = 0; r < n_; ++r){
        cur_index[r] = ret.rowptr_[r];
    }

    // fill colidx
    for(label r = 0; r < n_; ++r){
        for(label index = rowptr_[r]; index < rowptr_[r + 1]; ++index){
            label c = colidx_[index];
            if(c < r){
                ret.colidx_[cur_index[r]] = c;
                cur_index[r] += 1;
            }
        }
    }

    // check
    for(label r = 0; r < n_; ++r){
        assert(cur_index[r] == ret.rowptr_[r + 1]);
    }
    return ret;
}

csrPattern csrPattern::indirect_conflict(){
    csrPattern ret;

    ret.n_ = n_;

    // compute row_count
    std::vector<label> row_count(n_, 0);
    std::vector<label> mark(n_, -1);
    for(label r = 0; r < n_; ++r){
        for(label idx = rowptr_[r]; idx < rowptr_[r+1]; ++idx){
            label c = colidx_[idx];
            mark[c] = r;
        }
        
        for(label rr = 0; rr < n_; ++rr){
            if(rr == r) continue;
            for(label oidx = rowptr_[rr]; oidx < rowptr_[rr+1]; ++oidx){
                label cc = colidx_[oidx];
                if(mark[cc] == r){
                    row_count[r] += 1;
                }
            }
        }
    }

    // rowptr nnz
    ret.rowptr_.resize(n_ + 1);
    ret.rowptr_[0] = 0;
    for(label r = 0; r < n_; ++r){
        ret.rowptr_[r + 1] = ret.rowptr_[r] + row_count[r];
    }
    ret.nnz_ = ret.rowptr_[n_];
    ret.colidx_.resize(ret.nnz_);

    std::vector<label> cur_index(n_);
    for(label r = 0; r < n_; ++r){
        cur_index[r] = ret.rowptr_[r];
    }

    // reset mark
    for(label r = 0; r < n_; ++r){
        mark[r] = -1;
    }

    // fill colidx
    for(label r = 0; r < n_; ++r){
        for(label idx = rowptr_[r]; idx < rowptr_[r+1]; ++idx){
            label c = colidx_[idx];
            mark[c] = r;
        }
        for(label rr = 0; rr < n_; ++rr){
            if(rr == r) continue;
            for(label oidx = rowptr_[rr]; oidx < rowptr_[rr+1]; ++oidx){
                label cc = colidx_[oidx];
                if(mark[cc] == r){
                    ret.colidx_[cur_index[r]] = rr;
                    cur_index[r] += 1;
                }
            }
        }
    }

    // check
    for(label r = 0; r < n_; ++r){
        assert(cur_index[r] == ret.rowptr_[r + 1]);
    }
    return ret;
}

csrPattern operator+(const csrPattern& a, const csrPattern& b){
    if(a.n_ != b.n_){
        SeriousError << "In csrPattern friend operator+, the n_ of a and b have to be same !!!" << endl;
        SeriousError << "a.n_ : " << a.n_ << endl;
        SeriousError << "b.n_ : " << b.n_ << endl;
        MPI_Abort(PstreamGlobals::MPI_COMM_FOAM, -1);
    }

    csrPattern ret;
    ret.n_ = a.n_;

    // compute row_count
    std::vector<label> row_count(ret.n_, 0);
    std::vector<label> mark(ret.n_, -1);
    for(label r = 0; r < ret.n_; ++r){
        for(label idx = a.rowptr_[r]; idx < a.rowptr_[r+1]; ++idx){
            label c = a.colidx_[idx];
            mark[c] = r;
        }
        for(label idx = b.rowptr_[r]; idx < b.rowptr_[r+1]; ++idx){
            label c = b.colidx_[idx];
            mark[c] = r;
        }
        for(label c = 0; c < ret.n_; ++c){
            if(mark[c] == r){
                row_count[r] += 1;
            }
        }
    }

    // rowptr nnz
    ret.rowptr_.resize(ret.n_ + 1);
    ret.rowptr_[0] = 0;
    for(label r = 0; r < ret.n_; ++r){
        ret.rowptr_[r + 1] = ret.rowptr_[r] + row_count[r];
    }
    ret.nnz_ = ret.rowptr_[ret.n_];
    ret.colidx_.resize(ret.nnz_);

    std::vector<label> cur_index(ret.n_);
    for(label r = 0; r < ret.n_; ++r){
        cur_index[r] = ret.rowptr_[r];
    }

    // reset mark
    for(label r = 0; r < ret.n_; ++r){
        mark[r] = -1;
    }

    // fill colidx
    for(label r = 0; r < ret.n_; ++r){
        for(label idx = a.rowptr_[r]; idx < a.rowptr_[r+1]; ++idx){
            label c = a.colidx_[idx];
            mark[c] = r;
        }
        for(label idx = b.rowptr_[r]; idx < b.rowptr_[r+1]; ++idx){
            label c = b.colidx_[idx];
            mark[c] = r;
        }
        for(label c = 0; c < ret.n_; ++c){
            if(mark[c] == r){
                ret.colidx_[cur_index[r]] = c;
                cur_index[r] += 1;
            }
        }
    }
    
    // check
    for(label r = 0; r < ret.n_; ++r){
        assert(cur_index[r] == ret.rowptr_[r + 1]);
    }
    return ret;
}

void csrPattern::write_mtx(const std::string& filename) const {
    int mpisize, mpirank;
    int flag_mpi_init;
    MPI_Initialized(&flag_mpi_init);

    if(flag_mpi_init || !flag_mpi_init){
        MPI_Comm_rank(PstreamGlobals::MPI_COMM_FOAM, &mpirank);
        MPI_Comm_size(PstreamGlobals::MPI_COMM_FOAM, &mpisize);
    }
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
        fprintf(fw, "%ld %ld %ld\n", n_, n_, nnz_);
        for(label r = 0; r < n_; ++r){
            for(label index = rowptr_[r]; index < rowptr_[r+1]; ++index){
                label c = colidx_[index];
                fprintf(fw, "%ld %ld\n", r + 1, c + 1);
            }
        }
        fclose(fw);
    }
}

void csrPattern::order_natrue(std::vector<label>& order){
    order.resize(n_);
    for(label i = 0; i < n_; ++i){
        order[i] = i;
    }
}

void csrPattern::order_longest_row(std::vector<label>& order){
    std::vector<std::tuple<label,label>> row_length_tuples(n_);
    for(label r = 0; r < n_; ++r){
        row_length_tuples[r] = std::make_tuple(r, rowptr_[r+1] - rowptr_[r]);
    }
    // 按 行长 从大到小 行号 从小到大 排序
    std::sort(row_length_tuples.begin(), row_length_tuples.end(), [](const auto& a, const auto& b){
        return std::get<1>(a) == std::get<1>(b) ? std::get<0>(a) < std::get<0>(b) : std::get<1>(a) > std::get<1>(b);
    });
    order.resize(n_);
    for(label r = 0; r < n_; ++r){
        order[r] = std::get<0>(row_length_tuples[r]);
    }
}

void csrPattern::coloring(){
    Info << "coloring start" << endl;
    max_color_ = 0;

    color_.resize(n_);
    for(label i = 0; i < n_; ++i){
        color_[i] = -1;
    }

    std::vector<label> travel_order;
    // order_natrue(travel_order);
    order_longest_row(travel_order);


    std::vector<label> mark(n_, -1);

    for(label i = 0; i < n_; ++i){
        label u = travel_order[i];
        for(label index = rowptr_[u]; index < rowptr_[u+1]; ++index){
            label v = colidx_[index];
            label c = color_[v];
            if(c != -1){
                // i的邻居已经用过颜色c了
                mark[c] = i;
            }
        }
        label c = 0;
        while(c < max_color_ && mark[c] == i){
            c += 1;
        }
        if(c == max_color_){
            max_color_ += 1;
        }
        color_[u] = c;
    }

    for(label i = 0; i < n_; ++i){
        assert(color_[i] >= 0);
        assert(color_[i] < max_color_);
    }

    Info << "coloring end" << endl;
    Info << "max color : " << max_color_ << endl;

    color_count_.resize(max_color_);
    for(label c = 0; c < max_color_; ++c){
        color_count_[c] = 0;
    }
    for(label i = 0; i < n_; ++i){
        color_count_[color_[i]] += 1;
    }

    Info << "color count : " << endl;
    for(label c = 0; c < max_color_; ++c){
        Info << c << " : " << color_count_[c] << endl;
    }
}



}
