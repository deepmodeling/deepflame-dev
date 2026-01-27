#pragma once
#include <labelList.H>
#include <fvMesh.H>

namespace Foam{

class SubMesh{
public:

    enum PatchDirection {LEFT, UPPER, RIGHT, DOWN};

    int thread_id_;

    int thread_rid_;
    int thread_cid_;

    int thread_row_;
    int thread_col_;

    label nLocalCells_;
    label nAllCells_;
    label nLocalFaces_;
    label nLocalPatches_ = 4;
    label localPatchSize_[4]; // nLocalPatches_
    // scalar* localPatch_[4];  // left upper right down;
    labelList localGlabalFacePtr_; // 每个local cell的 global face id
    labelList localGlabalFace_; // 每个local cell的 global face id

public:

    SubMesh(){};

    label nLocalCells() const {return nLocalCells_;};
    label nAllCells() const {return nAllCells_;};
    label nLocalFaces() const {return nLocalFaces_;};
    label nLocalPatches() const {return nLocalPatches_;};
    const label* localPatchSize() const {return localPatchSize_;};
    label localPatchSize(label i) const {return localPatchSize_[i];};
    const labelUList& localGlabalFacePtr() const {return localGlabalFacePtr_;};
    const labelUList& localGlabalFace() const {return localGlabalFace_;};

    bool hasRightHalo() const {
        return (thread_cid_ + 1) < thread_col_;
    }
    bool hasDownHalo() const {
        return (thread_rid_ + 1) < thread_row_;
    }
    bool hasLeftHalo() const {
        return thread_cid_ > 0;
    }
    bool hasUpperHalo() const {
        return thread_rid_ > 0;
    }

    int leftThreadRank() const {
        return thread_id_ - 1;
    }

    int rightThreadRank() const {
        return thread_id_ + 1;
    }

    int upperThreadRank() const {
        return thread_id_ - thread_col_;
    }

    int downThreadRank() const {
        return thread_id_ + thread_col_;
    }

    virtual ~SubMesh(){
        // delete [] localPatch_[LEFT];
        // delete [] localPatch_[UPPER];
        // delete [] localPatch_[RIGHT];
        // delete [] localPatch_[DOWN];
    };

    
};

}








