#pragma once

#include "StructuredMeshSchedule.H"

namespace Foam{

// 3D结构化网格z轴1D分块图染色

class XYBlock1DColoringStructuredMeshSchedule : public StructuredMeshSchedule{
protected:

    XYBlock1DColoringStructuredMeshSchedule(const fvMesh& mesh);

    labelList cell_scheduling_;
    labelList face_scheduling_;
public:

    virtual ~XYBlock1DColoringStructuredMeshSchedule(){};

    const labelList& cell_scheduling() const {return cell_scheduling_;};
    const labelList& face_scheduling() const {return face_scheduling_;};

    virtual void show() const {
        MeshSchedule::show();
        Info << "XYBlock1DColoringStructuredMeshSchedule show ------------" << endl;
        Info << "---------------------------------------------" << endl;
    };

    static XYBlock1DColoringStructuredMeshSchedule* xyBlock1DColoringStructuredMeshSchedule_;

    static void buildXYBlock1DColoringStructuredMeshSchedule(const fvMesh& mesh);
    
    static const XYBlock1DColoringStructuredMeshSchedule& getXYBlock1DColoringStructuredMeshSchedule() {
        assert(xyBlock1DColoringStructuredMeshSchedule_ != nullptr); 
        return *xyBlock1DColoringStructuredMeshSchedule_;
    };
};


}
