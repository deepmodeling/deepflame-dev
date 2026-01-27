#include "schedule.H"


namespace Foam{

const MeshSchedule& getSchedule(){

    return MeshSchedule::getMeshSchedule();

    // #if defined(OPT_FACE2CELL_COLORING_SCHEDULE)

    // return XYBlock1DColoringStructuredMeshSchedule::getXYBlock1DColoringStructuredMeshSchedule();
    
    // #elif defined(OPT_FACE2CELL_PARTITION)
    
    // return XBlock2DPartitionStructuredMeshSchedule::getXBlock2DPartitionStructuredMeshSchedule();
    
    // #else
    // #error "getSchedule : OPT_FACE2CELL_COLORING_SCHEDULE OPT_FACE2CELL_PARTITION"
    // #endif
}

}