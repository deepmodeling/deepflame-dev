#pragma once
#include <fvMesh.H>
#include <vector>
#include <utility>
#include "csrPattern.H"

#define bid2d(rbid, cbid) (rbid * rowBlockCount_ + cbid)

namespace Foam{

class BlockPattern{
private:
    
    label n_;

    label rowBlockCount_;
    labelList rowBlockPtr_;


    std::vector<std::vector<std::pair<label, label>>> blocks_;
    


public:

    BlockPattern(const csrPattern csr, const labelList& regionPtr);

    ~BlockPattern(){};

    void show() const;

};

}
