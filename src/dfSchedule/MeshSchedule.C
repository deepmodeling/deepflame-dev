#include "MeshSchedule.H"

namespace Foam{

MeshSchedule::MeshSchedule(const fvMesh& mesh) : mesh_(mesh){
    Info << "MeshSchedule::MeshSchedule start" << endl;
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


    // reverse Mesh
    const labelUList& facePtr_ = facePtr();
    const labelUList& neighbour_ = neighbour();
    const labelUList& owner_ = mesh_.owner();

    labelList reverseFaceCount(nCells_, 0);

    for(label f = 0; f < nFaces_; ++f){
        reverseFaceCount[neighbour_[f]] += 1;
    }

    reverseFacePtr_.resize(nCells_ + 1);
    reverseNeighbour_.resize(nFaces_);
    faceIndexMapping_.resize(nFaces_);

    labelList reverseFaceIndex(nCells_ + 1);

    reverseFacePtr_[0] = 0;
    reverseFaceIndex[0] = 0;
    for(label c = 1; c <= nCells_; ++c){
        reverseFacePtr_[c] = reverseFacePtr_[c - 1] + reverseFaceCount[c - 1];
        reverseFaceIndex[c] = reverseFaceIndex[c - 1] + reverseFaceCount[c - 1];
    }
    
    assert(reverseFacePtr_[nCells_] == nFaces_);

    for(label c = 0; c < nCells_; ++c){
        for(label f = facePtr_[c]; f < facePtr_[c+1]; ++f){
            label own = owner_[f];
            label nei = neighbour_[f];
            assert(own == c);
            label index = reverseFaceIndex[nei];
            reverseNeighbour_[index] = c;
            faceIndexMapping_[index] = f;
            reverseFaceIndex[nei] += 1;
        }
    }

    Info << "MeshSchedule::MeshSchedule end" << endl;
}


}
