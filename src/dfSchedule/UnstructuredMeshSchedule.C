#include "UnstructuredMeshSchedule.H"
#include "csrPattern.H"
#include <cassert>

namespace Foam{

UnstructuredMeshSchedule::UnstructuredMeshSchedule(const fvMesh& mesh):MeshSchedule(mesh){}

}
