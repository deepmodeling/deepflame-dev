#pragma once
#include <fvMesh.H>
#include "SubMesh.H"


namespace Foam{

class PartitionMeshSchedule{
protected:

    int partition_count_;

    PtrList<SubMesh> subMeshList_;

    labelList belongTo_;  // nCells_

public:

    PartitionMeshSchedule(){};
    ~PartitionMeshSchedule(){};

    const SubMesh& subMesh(label tid) const {return subMeshList_[tid];};

    label belongTo(label gc) const {
        return belongTo_[gc];
    }


};

}

