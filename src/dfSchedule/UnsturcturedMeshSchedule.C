#include "UnstructuredMeshSchedule.H"
#include "csrPattern.H"
#include <cassert>

namespace Foam{

void UnstructuredMeshSchedule::buildUnstructuredMeshSchedule(const fvMesh& mesh){
    if(meshSchedulePtr_ == nullptr){
        meshSchedulePtr_ = new UnstructuredMeshSchedule(mesh);
    }else{
        Info << "meshSchedule has been constructed !!!" << endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
}
    
UnstructuredMeshSchedule::UnstructuredMeshSchedule(const fvMesh& mesh):MeshSchedule(mesh){}

}
