#pragma once

#include "SubMesh.H"

namespace Foam{

class StructuredSubMesh : public SubMesh{
public:

    // global mesh info
    label x_dim_; 
    label y_dim_;
    label z_dim_;

    // Structured 2D partition
    label x_len_;
    
    // row
    label z_start_;  
    label z_end_;
    label z_len_;
    
    // col
    label y_start_;
    label y_end_;
    label y_len_;


public:

    // x_len_ * y_len_ * z_len_             local cell
    label localIndex(label lx, label ly, label lz) const {
        return lx + ly * x_len_ + lz * x_len_ * y_len_;
    }

    // x_len_ * (y_len_ + 1) * (z_len_ + 1) local + halo cell
    label localHaloIndex(label lx, label ly, label lz) const {
        return lx + ly * x_len_ + lz * x_len_ * (y_len_ + 1);
    }

    label globalX(label lx) const {return lx;};
    label globalY(label ly) const {return ly + y_start_;};
    label globalZ(label lz) const {return lz + z_start_;};

    label globalIndex(label gx, label gy, label gz) const {
        return gx + gy * x_dim_ + gz * x_dim_ * y_dim_;
    }

    label localZLen() const {return z_len_;};
    label localYLen() const {return y_len_;};
    label localXLen() const {return x_len_;};

    label globalZLen() const {return z_dim_;};
    label globalYLen() const {return y_dim_;};
    label globalXLen() const {return x_dim_;};


    StructuredSubMesh(){};

    virtual ~StructuredSubMesh() = default;


    void check(const fvMesh& mesh);

};

}
