#pragma once

#include "MeshSchedule.H"
#include <vector>
#include <tuple>

namespace Foam{

// 3D结构化网格分块图染色

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
        Info << "x_dim : " << x_dim_ << endl;
        Info << "y_dim : " << y_dim_ << endl;
        Info << "z_dim : " << z_dim_ << endl;
        Info << "---------------------------------------------" << endl;
    };

    label index(label x, label y, label z) const;
    
    void xyz(label index, label& x, label& y, label& z) const;

    static void buildStructuredMeshSchedule(const fvMesh& mesh);
    
    static const StructuredMeshSchedule& getStructuredMeshSchedule() {
        assert(meshSchedulePtr_ != nullptr); 
        return dynamic_cast<StructuredMeshSchedule&>(*meshSchedulePtr_);
    };
};



}
