#pragma once

#include "SubMesh.H"

namespace Foam{

class UnstructuredSubMesh : public SubMesh{
public:

    // Unstructured partition
    labelList localOwner_;  // nLocalFaces_
    labelList localNeighbour_; // nLocalFaces_
    labelList localFacePtr_; // nLocalCells_

    // local id to global id
    labelList cellMappingForward_; // nLocalCells_ + nGhostCells_ 
    // global id to local id
    labelList cellMappingBackward_; // nCell

public:

    UnstructuredSubMesh(){};

    virtual ~UnstructuredSubMesh() = default;

    
};

}
