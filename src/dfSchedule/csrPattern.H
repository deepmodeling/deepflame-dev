#pragma once
#include <fvMesh.H>
#include <vector>

namespace Foam{

class csrPattern{
private:
    label n_;
    label nnz_;
    labelList rowptr_;
    labelList colidx_;

    labelList color_;
    label max_color_;
    labelList color_count_;

    void order_natrue(std::vector<label>& order);
    void order_longest_row(std::vector<label>& order);

public:
    
    csrPattern():n_(0),nnz_(0){};

    csrPattern(const fvMesh& mesh);

    csrPattern(const csrPattern& other):
        n_(other.n_),nnz_(other.nnz_),
        rowptr_(other.rowptr_), colidx_(other.colidx_), 
        color_(other.color_), max_color_(other.max_color_), color_count_(other.color_count_){
        Info << "Warning : csrPattern::copy constructor is called !!!" << endl;
    };

    ~csrPattern(){};

    void coloring();

    label color(label index){
        return color_[index];
    }

    label max_color(){
        return max_color_;
    }

    csrPattern blocking(label block_size);

    csrPattern lower_part();

    csrPattern indirect_conflict();

    friend csrPattern operator+(const csrPattern& a, const csrPattern& b);

    void write_mtx(const std::string& filename) const;
};


}
