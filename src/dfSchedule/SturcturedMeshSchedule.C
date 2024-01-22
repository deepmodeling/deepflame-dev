#include "StructuredMeshSchedule.H"
#include "csrPattern.H"
#include <cassert>

namespace Foam{

void StructuredMeshSchedule::buildStructuredMeshSchedule(const fvMesh& mesh){
    if(meshSchedulePtr_ == nullptr){
        meshSchedulePtr_ = new StructuredMeshSchedule(mesh);
    }else{
        Info << "meshSchedule has been constructed !!!" << endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
}
    
label StructuredMeshSchedule::index(label x, label y, label z) const {
    return z * x_dim_ * y_dim_ + y * x_dim_ + x;
}

void StructuredMeshSchedule::xyz(label index, label& x, label& y, label& z) const {
    x = index % x_dim_;
    index = index / x_dim_;
    y = index % y_dim_;
    index = index / y_dim_;
    z = index;
}

StructuredMeshSchedule::StructuredMeshSchedule(const fvMesh& mesh):MeshSchedule(mesh){
    const labelUList& face_ptr = mesh.lduAddr().ownerStartAddr();
    const labelUList& l = mesh.lduAddr().lowerAddr();
    const labelUList& u = mesh.lduAddr().upperAddr();

    std::vector<std::tuple<label,label,label>> points;

    const auto& cellCentres = mesh.cellCentres();
    forAll(cellCentres, i){
        points.push_back(std::make_tuple(static_cast<label>(cellCentres[i][0] * 1e6), static_cast<label>(cellCentres[i][1] * 1e6), static_cast<label>(cellCentres[i][2] * 1e6)));
    }

    // compute x y z dim
    x_dim_ = 1;
    for(size_t i = 1; i < points.size(); ++i){
        if(std::get<1>(points[i]) == std::get<1>(points[i-1])){
            x_dim_ += 1;
        }else{
            break;
        }
    }

    y_dim_ = 1;
    for(size_t i = 1; i < points.size(); ++i){
        if(std::get<2>(points[i]) == std::get<2>(points[i-1])){
            y_dim_ += 1;
        }else{
            break;
        }
    }
    
    y_dim_ /= x_dim_;
    z_dim_ = nCells_ / y_dim_ / x_dim_;

    // simple layer scheduling
    cell_scheduling_.resize(z_dim_ + 1);
    face_scheduling_.resize(z_dim_ + 1);

    cell_scheduling_[0] = 0;
    face_scheduling_[0] = 0;

    for(label i = 0 ; i < z_dim_; i++){
        cell_scheduling_[i+1] = cell_scheduling_[i] + (x_dim_ * y_dim_);
        face_scheduling_[i+1] = face_ptr[cell_scheduling_[i+1]];
    }

    // physics solver co-design
    // renumber && coloring 
    
    // csrPattern meshPattern(mesh);
    // meshPattern.coloring();
    // meshPattern.write_mtx("sparse_pattern");

    // csrPattern block32 = meshPattern.blocking(32);
    // block32.write_mtx("sparse_pattern_block_32");

    // csrPattern yzdim_graph = meshPattern.blocking(x_dim_);
    // yzdim_graph.write_mtx("sparse_pattern_block_xdim_" + std::to_string(x_dim_));

    // csrPattern yzdim_lower_part_graph = yzdim_graph.lower_part();
    // yzdim_lower_part_graph.write_mtx("sparse_pattern_block_xdim_lower_part");

    // csrPattern yzdim_indirect_graph = yzdim_lower_part_graph.indirect_conflict();
    // yzdim_indirect_graph.write_mtx("sparse_pattern_block_xdim_indirect_graph");

    // csrPattern yzdim_symv_conflict_graph = yzdim_graph + yzdim_indirect_graph;
    // yzdim_symv_conflict_graph.write_mtx("sparse_pattern_block_xdim_symv_conflict_graph");
    // yzdim_symv_conflict_graph.coloring();

    // csrPattern zdim_graph = meshPattern.blocking(x_dim_ * y_dim_);
    // zdim_graph.write_mtx("sparse_pattern_block_xydim_" + std::to_string(x_dim_ * y_dim_));
}

}
