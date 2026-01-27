#pragma once

#include <fvMesh.H>
#include <cassert>
#include <mpi.h>

namespace Foam{


class MeshSchedule{
public:
    enum PatchType {wall, processor};

protected:
    const fvMesh& mesh_;

    label nCells_;
    label nFaces_;
    label nPatches_;
    List<PatchType> patchTypes_; // nPatches_
    labelList patchSizes_; // nPatches_

    labelList reverseFacePtr_; // nCells + 1
    labelList reverseNeighbour_; // nFaces
    labelList faceIndexMapping_; // reverseFaceIndex -> faceIndex; // nFaces

    MeshSchedule(const fvMesh& mesh);

public:

    virtual ~MeshSchedule(){};

    label nCells() const {return nCells_;};
    label nFaces() const {return nFaces_;};
    label nPatches() const {return nPatches_;};
    const List<PatchType>& patchTypes() const {return patchTypes_;};
    const labelUList& patchSizes() const {return patchSizes_;};
    
    const labelUList& facePtr() const {return mesh_.lduAddr().ownerStartAddr();};
    const labelUList& neighbour() const {return mesh_.neighbour();};
    const labelUList& reverseFacePtr() const {return reverseFacePtr_;};
    const labelUList& reverseNeighbour() const {return reverseNeighbour_;};
    const labelUList& faceIndexMapping() const {return faceIndexMapping_;};

    static MeshSchedule* meshSchedule_;

    static void buildMeshSchedule(const fvMesh& mesh);

    virtual void show() const {
        Info << "MeshSchedule show ---------------------------" << endl;
        Info << "nCells : " << nCells_ << endl;
        Info << "nFaces : " << nFaces_ << endl;
        Info << "nPatches : " << nPatches_ << endl;
        Info << "patchSizes_ : " << patchSizes_ << endl;
        Info << "---------------------------------------------" << endl;
    };

    static const MeshSchedule& getMeshSchedule() {
        assert(meshSchedule_ != nullptr); 
        return *meshSchedule_;
    };

};


}
