#pragma once

#include "MeshSchedule.H"
#include <vector>
#include <tuple>

namespace Foam{

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

};


}
