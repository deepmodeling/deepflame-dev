#pragma once

#include "StructuredMeshSchedule.H"
#include "PartitionMeshSchedule.H"
#include "StructuredSubMesh.H"

namespace Foam{



class XBlock2DPartitionStructuredMeshSchedule : public StructuredMeshSchedule, public PartitionMeshSchedule{
protected:

    label partition_row_;
    label partition_col_;

    XBlock2DPartitionStructuredMeshSchedule(const fvMesh& mesh);

public:

    virtual ~XBlock2DPartitionStructuredMeshSchedule(){};

    virtual void show() const {
        MeshSchedule::show();
        Info << "XBlock2DPartitionStructuredMeshSchedule show --" << endl;
        Info << "---------------------------------------------" << endl;
    };

    static XBlock2DPartitionStructuredMeshSchedule* xBlock2DPartitionStructuredMeshSchedule_;

    static void buildXBlock2DPartitionStructuredMeshSchedule(const fvMesh& mesh);
    
    static const XBlock2DPartitionStructuredMeshSchedule& getXBlock2DPartitionStructuredMeshSchedule() {
        assert(xBlock2DPartitionStructuredMeshSchedule_ != nullptr); 
        return *xBlock2DPartitionStructuredMeshSchedule_;
    };
};




}
