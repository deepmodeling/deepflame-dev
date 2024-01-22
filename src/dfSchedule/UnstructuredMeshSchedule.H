#pragma once

#include "MeshSchedule.H"
#include <vector>
#include <tuple>

namespace Foam{

// 3D结构化网格分块图染色

class UnstructuredMeshSchedule : public MeshSchedule{
protected:

    UnstructuredMeshSchedule(const fvMesh& mesh);

public:

    virtual ~UnstructuredMeshSchedule(){};

    virtual void show() const {
        MeshSchedule::show();
        Info << "UnstructuredMeshSchedule show ------------------" << endl;
        Info << "---------------------------------------------" << endl;
    };


    static void buildUnstructuredMeshSchedule(const fvMesh& mesh);
    
    static const UnstructuredMeshSchedule& getUnstructuredMeshSchedule() {
        assert(meshSchedulePtr_ != nullptr); 
        return dynamic_cast<UnstructuredMeshSchedule&>(*meshSchedulePtr_);
    };
};



}
