#include "StructuredMeshSchedule.H"
#include "csrPattern.H"
#include <cassert>

namespace Foam{

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
    Info << "StructuredMeshSchedule::StructuredMeshSchedule start" << endl;
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
    Info << "StructuredMeshSchedule::StructuredMeshSchedule end" << endl;
}

}
