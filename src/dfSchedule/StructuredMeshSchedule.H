#pragma once

#include "MeshSchedule.H"
#include <vector>
#include <tuple>

namespace Foam{

class StructuredMeshSchedule : public MeshSchedule{
protected:

    label x_dim_;
    label y_dim_;
    label z_dim_;

    StructuredMeshSchedule(const fvMesh& mesh);

public:

    virtual ~StructuredMeshSchedule(){};

    virtual void show() const {
        MeshSchedule::show();
        Info << "StructuredMeshSchedule show ------------------" << endl;
        Info << "x_dim_ : " << x_dim_ << endl;
        Info << "y_dim_ : " << y_dim_ << endl;
        Info << "z_dim_ : " << z_dim_ << endl;
        Info << "---------------------------------------------" << endl;
    };

    label index(label x, label y, label z) const;
    
    void xyz(label index, label& x, label& y, label& z) const;

};

}