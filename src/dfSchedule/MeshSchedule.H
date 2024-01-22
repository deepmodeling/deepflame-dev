#pragma once

#include <fvMesh.H>
#include <cassert>
#include <mpi.h>

namespace Foam{


class MeshSchedule{
public:
    enum PatchType {wall, processor};

protected:
    label nCells_;
    label nFaces_;
    label nPatches_;
    List<PatchType> patchTypes_; // nPatches_
    labelList patchSizes_; // nPatches_

    labelList cell_scheduling_;
    labelList face_scheduling_;

    static MeshSchedule* meshSchedulePtr_; // 单例

    MeshSchedule(const fvMesh& mesh);

public:

    virtual ~MeshSchedule(){};

    label nCells() const {return nCells_;};
    label nFaces() const {return nFaces_;};
    label nPatches() const {return nPatches_;};
    const List<PatchType>& patchTypes() const {return patchTypes_;};
    const labelList& patchSizes() const {return patchSizes_;};

    const labelList& cell_scheduling() const {return cell_scheduling_;};
    const labelList& face_scheduling() const {return face_scheduling_;};

    virtual void show() const {
        Info << "MeshSchedule show ---------------------------" << endl;
        Info << "nCells : " << nCells_ << endl;
        Info << "nFaces : " << nFaces_ << endl;
        Info << "nPatches : " << nPatches_ << endl;
        Info << "patchSizes_ : " << patchSizes_ << endl;
        Info << "---------------------------------------------" << endl;
    };

    // static void buildMeshSchedule(const fvMesh& mesh);

    static const MeshSchedule& getMeshSchedule() {assert(meshSchedulePtr_ != nullptr); return *meshSchedulePtr_;};
};



}
