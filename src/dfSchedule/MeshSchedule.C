#include "MeshSchedule.H"

namespace Foam{

MeshSchedule* MeshSchedule::meshSchedulePtr_ = nullptr;

MeshSchedule::MeshSchedule(const fvMesh& mesh){
    nCells_ = mesh.nCells();
    nFaces_ = mesh.neighbour().size();
    nPatches_ = mesh.boundary().size();
    
    patchTypes_.resize(nPatches_);
    patchSizes_.resize(nPatches_);
    forAll(mesh.boundary(), patchi){
        const fvPatch& patch = mesh.boundary()[patchi];
        // const labelUList& pFaceCells = patch.faceCells();
        patchSizes_[patchi] = patch.size();
        if(patch.type() == "wall"){
            patchTypes_[patchi] = PatchType::wall;
        }else if(patch.type() == "processor"){
            patchTypes_[patchi] = PatchType::processor;
        }else{
            Info << "patch.type() : " << patch.type() << "is not supported !!!" << endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
}


}
