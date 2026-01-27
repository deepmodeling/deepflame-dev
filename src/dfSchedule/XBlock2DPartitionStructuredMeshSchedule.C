#include "XBlock2DPartitionStructuredMeshSchedule.H"
#include <cassert>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <mpi.h>
#include "env.H"
#include <cassert>


namespace Foam{

XBlock2DPartitionStructuredMeshSchedule* XBlock2DPartitionStructuredMeshSchedule::xBlock2DPartitionStructuredMeshSchedule_ = nullptr;

void XBlock2DPartitionStructuredMeshSchedule::buildXBlock2DPartitionStructuredMeshSchedule(const fvMesh& mesh){
    if(xBlock2DPartitionStructuredMeshSchedule_ == nullptr){
        xBlock2DPartitionStructuredMeshSchedule_ = new XBlock2DPartitionStructuredMeshSchedule(mesh);
    }else{
        Info << "meshSchedule has been constructed !!!" << endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
}
    
XBlock2DPartitionStructuredMeshSchedule::XBlock2DPartitionStructuredMeshSchedule(const fvMesh& mesh):StructuredMeshSchedule(mesh){
    Info << "XBlock2DPartitionStructuredMeshSchedule::XBlock2DPartitionStructuredMeshSchedule start" << endl;
#ifdef _OPENMP
    partition_count_ = env::get_int_default("MESH_PARTITION_COUNT", omp_get_max_threads());
#else
    partition_count_ = env::get_int_default("MESH_PARTITION_COUNT", 1);
#endif

    if(partition_count_ == 1){
        Info << "Waring : partition_count_ is equal to 1. Mesh division is unnecessary !!!" << endl;
    }

    // spcific for 12 for fugaku, 64 for sw
    // factoring is better
    if(partition_count_ == 12){ // for fugaku
        partition_row_ = 3;
        partition_col_ = 4;
    }else if(partition_count_ == 64){ // for sw
        partition_row_ = 8;
        partition_col_ = 8;
    }else{
        Info << "Error : partition_count_ == " << partition_count_ << " is not supported now !!!" << endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    const labelUList facePtr_ = facePtr();
    const labelUList neighbour_ = neighbour();
    const labelUList& owner_ = mesh_.owner();

    // partition cell
    belongTo_.resize(nCells_, -1);
    subMeshList_.resize(partition_count_);
    
    // local cell
    for(int p = 0; p < partition_count_; ++p){
        subMeshList_.set(p, new StructuredSubMesh());
        StructuredSubMesh& subMesh = dynamic_cast<StructuredSubMesh&>(subMeshList_[p]);

        subMesh.thread_id_ = p;

        subMesh.thread_row_ = partition_row_;
        subMesh.thread_col_ = partition_col_;

        subMesh.thread_rid_ = p / partition_col_;
        subMesh.thread_cid_ = p % partition_col_;      
        
        subMesh.x_dim_ = x_dim_;
        subMesh.y_dim_ = y_dim_;
        subMesh.z_dim_ = z_dim_;

        subMesh.x_len_ = x_dim_;
        subMesh.z_start_ = (subMesh.thread_rid_ * z_dim_) / partition_row_;
        subMesh.z_end_ = ((subMesh.thread_rid_ + 1) * z_dim_) / partition_row_;
        subMesh.z_len_ = subMesh.z_end_ - subMesh.z_start_;
        subMesh.y_start_ = (subMesh.thread_cid_ * y_dim_) / partition_col_;
        subMesh.y_end_ = ((subMesh.thread_cid_ + 1) * y_dim_) / partition_col_;
        subMesh.y_len_ = subMesh.y_end_ - subMesh.y_start_;

        subMesh.nLocalCells_ = subMesh.z_len_ * subMesh.y_len_ * subMesh.x_len_;
        // partition local cell
        for(label gz = subMesh.z_start_; gz < subMesh.z_end_; ++gz){
            for(label gy = subMesh.y_start_; gy < subMesh.y_end_; ++gy){
                for(label gx = 0; gx < x_dim_; ++gx){
                    label gc = subMesh.globalIndex(gx, gy, gz);
                    belongTo_[gc] = p;
                }
            }
        }
    }

    // local global face
    for(int p = 0; p < partition_count_; ++p){
        StructuredSubMesh& subMesh = dynamic_cast<StructuredSubMesh&>(subMeshList_[p]);

        subMesh.nAllCells_ = subMesh.x_len_ * (subMesh.y_len_ + 1) * (subMesh.z_len_ + 1);

        labelList localGlabalFaceCount(subMesh.nLocalCells_, 0);
        subMesh.nLocalFaces_ = 0;
        for(label gz = subMesh.z_start_; gz < subMesh.z_end_; ++gz){
            label lz = gz - subMesh.z_start_;
            for(label gy = subMesh.y_start_; gy < subMesh.y_end_; ++gy){
                label ly = gy - subMesh.y_start_;
                for(label gx = 0; gx < x_dim_; ++gx){
                    label lx = gx;
                    label lc = subMesh.localIndex(lx, ly, lz);
                    label gc = subMesh.globalIndex(gx, gy, gz);
                    for(label gf = facePtr_[gc]; gf < facePtr_[gc + 1]; ++gf){
                        localGlabalFaceCount[lc] += 1;
                        subMesh.nLocalFaces_ += 1;
                    }
                }
            }
        }

        // init ptr
        labelList localGlabalFaceIndex(subMesh.nLocalCells_ + 1);
        subMesh.localGlabalFacePtr_.resize(subMesh.nLocalCells_ + 1);

        localGlabalFaceIndex[0] = 0;
        subMesh.localGlabalFacePtr_[0] = 0;
        for(label lc = 1; lc <= subMesh.nLocalCells_; ++lc){
            localGlabalFaceIndex[lc] = localGlabalFaceIndex[lc - 1] + localGlabalFaceCount[lc - 1];
            subMesh.localGlabalFacePtr_[lc] = subMesh.localGlabalFacePtr_[lc - 1] + localGlabalFaceCount[lc - 1];
        }

        assert(subMesh.localGlabalFacePtr_[subMesh.nLocalCells_] = subMesh.nLocalFaces_);

        subMesh.localGlabalFace_.resize(subMesh.nLocalFaces_);

        // fill localGlabalFace
        for(label gz = subMesh.z_start_; gz < subMesh.z_end_; ++gz){
            label lz = gz - subMesh.z_start_;
            for(label gy = subMesh.y_start_; gy < subMesh.y_end_; ++gy){
                label ly = gy - subMesh.y_start_;
                for(label gx = 0; gx < x_dim_; ++gx){
                    label lx = gx;
                    label lc = subMesh.localIndex(lx, ly, lz);
                    label gc = subMesh.globalIndex(gx, gy, gz);

                    for(label gf = facePtr_[gc]; gf < facePtr_[gc + 1]; ++gf){
                        label index = localGlabalFaceIndex[lc];
                        subMesh.localGlabalFace_[index] = gf;
                        localGlabalFaceIndex[lc] += 1;
                    }
                }
            }
        }

        // init patch
        subMesh.localPatchSize_[SubMesh::PatchDirection::LEFT] = subMesh.x_len_ * subMesh.z_len_;
        subMesh.localPatchSize_[SubMesh::PatchDirection::RIGHT] = subMesh.x_len_ * subMesh.z_len_;
        subMesh.localPatchSize_[SubMesh::PatchDirection::UPPER] = subMesh.x_len_ * subMesh.y_len_;
        subMesh.localPatchSize_[SubMesh::PatchDirection::DOWN] = subMesh.x_len_ * subMesh.y_len_;

        // localPatch_[SubMesh::PatchDirection::LEFT] = new scalar[subMesh.localPatchSize_[SubMesh::PatchDirection::LEFT]];
        // localPatch_[SubMesh::PatchDirection::RIGHT] = new scalar[subMesh.localPatchSize_[SubMesh::PatchDirection::RIGHT]];
        // localPatch_[SubMesh::PatchDirection::UPPER] = new scalar[subMesh.localPatchSize_[SubMesh::PatchDirection::UPPER]];
        // localPatch_[SubMesh::PatchDirection::DOWN] = new scalar[subMesh.localPatchSize_[SubMesh::PatchDirection::DOWN]];

        subMesh.check(mesh);
    }

    // halo

    Info << "XBlock2DPartitionStructuredMeshSchedule::XBlock2DPartitionStructuredMeshSchedule end" << endl;
}

}
