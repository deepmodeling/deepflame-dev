#include "XYBlock1DColoringStructuredMeshSchedule.H"
#include "csrPattern.H"
#include <cassert>

namespace Foam{

XYBlock1DColoringStructuredMeshSchedule* XYBlock1DColoringStructuredMeshSchedule::xyBlock1DColoringStructuredMeshSchedule_ = nullptr;

void XYBlock1DColoringStructuredMeshSchedule::buildXYBlock1DColoringStructuredMeshSchedule(const fvMesh& mesh){
    if(xyBlock1DColoringStructuredMeshSchedule_ == nullptr){
        xyBlock1DColoringStructuredMeshSchedule_ = new XYBlock1DColoringStructuredMeshSchedule(mesh);
    }else{
        Info << "xyBlock1DColoringStructuredMeshSchedule has been constructed !!!" << endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
}
    
XYBlock1DColoringStructuredMeshSchedule::XYBlock1DColoringStructuredMeshSchedule(const fvMesh& mesh):StructuredMeshSchedule(mesh){
    const labelUList& face_ptr = mesh.lduAddr().ownerStartAddr();
    // simple z layer scheduling
    cell_scheduling_.resize(z_dim_ + 1);
    face_scheduling_.resize(z_dim_ + 1);

    cell_scheduling_[0] = 0;
    face_scheduling_[0] = 0;

    for(label i = 0 ; i < z_dim_; i++){
        cell_scheduling_[i+1] = cell_scheduling_[i] + (x_dim_ * y_dim_);
        face_scheduling_[i+1] = face_ptr[cell_scheduling_[i+1]];
    }
}

}
