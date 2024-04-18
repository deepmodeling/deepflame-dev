#include "StructuredSubMesh.H"
#include <cassert>

namespace Foam{


void StructuredSubMesh::check(const fvMesh& mesh){

    const labelUList& owner = mesh.owner();
    const labelUList& neighbour = mesh.neighbour();

    // check local cell
    for(label lz = 0; lz < z_len_; ++lz){
        label gz = globalZ(lz);
        for(label ly = 0; ly < y_len_; ++ly){
            label gy = globalY(ly);
            for(label lx = 0; lx < x_len_; ++lx){
                label gx = globalX(lx);
                label lc = localIndex(lx, ly, lz);
                label gc = globalIndex(gx, gy, gz);

                label localFaceCount = 0;

                label gfp = localGlabalFacePtr_[lc];
                // (x+1, y, z);
                if(gx + 1 < x_dim_){
                    localFaceCount += 1;
                    label gnc = globalIndex(gx + 1, gy, gz);
                    label gf = localGlabalFace_[gfp];
                    assert(gc == owner[gf]);
                    assert(gnc == neighbour[gf]);
                    gfp += 1;
                }
                // (x, y+1, z);
                if(gy + 1 < y_dim_){
                    localFaceCount += 1;
                    label gnc = globalIndex(gx, gy + 1, gz);
                    label gf = localGlabalFace_[gfp];
                    assert(gc == owner[gf]);
                    assert(gnc == neighbour[gf]);
                    gfp += 1;
                }
                // (x, y, z+1);
                if(gz + 1 < z_dim_){
                    localFaceCount += 1;
                    label gnc = globalIndex(gx, gy, gz + 1);
                    label gf = localGlabalFace_[gfp];
                    assert(gc == owner[gf]);
                    assert(gnc == neighbour[gf]);
                    gfp += 1;
                }
                assert(gfp == localGlabalFacePtr_[lc+1]);
                assert(localFaceCount == (localGlabalFacePtr_[lc+1] - localGlabalFacePtr_[lc]));
            }
        }
    }



}

}
