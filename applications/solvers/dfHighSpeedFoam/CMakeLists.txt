project(dfHighSpeedFoam LANGUAGES CXX)

include_directories(
    ${OpenFOAM_APP}/solvers/compressible/rhoCentralFoam/BCs/lnInclude
    ${OpenFOAM_SRC}/finiteVolume/cfdTools 
    ${OpenFOAM_SRC}/finiteVolume/lnInclude 
    ${OpenFOAM_SRC}/transportModels/compressible/lnInclude 
    ${OpenFOAM_SRC}/thermophysicalModels/basic/lnInclude 
    ${OpenFOAM_SRC}/TurbulenceModels/turbulenceModels/lnInclude 
    ${OpenFOAM_SRC}/TurbulenceModels/compressible/lnInclude
    ${OpenFOAM_SRC}/meshTools/lnInclude 
    ${OpenFOAM_SRC}/sampling/lnInclude 
    ${OpenFOAM_SRC}/dynamicFvMesh/lnInclude 
    ${OpenFOAM_SRC}/Pstream/mpi 
    ${dfcantera_inc}
    ${dfchemistry_inc}
    ${dfcombustion_inc}
    ${CANTERA_ROOT}/include 
    ${MPI_INCLUDE_PATH}
    ${PROJECT_SOURCE_DIR}
)


add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/dfHighSpeedFoam.C)

if(DEFINED ENV{PYTHON_INC_DIR})
  add_definitions(-DUSE_PYTORCH)
  include_directories(
  /usr/include/python3.8
  /home/runze/.local/lib/python3.8/site-packages/pybind11/include
  )
  target_link_libraries(${PROJECT_NAME} /usr/lib/python3.8/config-3.8-x86_64-linux-gnu/libpython3.8.so)
endif()

target_link_libraries(${PROJECT_NAME} $ENV{FOAM_LIBBIN}/libfiniteVolume.so libmeshTools.so libcompressibleTransportModels.so libturbulenceModels.so libOpenFOAM.so libdynamicFvMesh.so libtopoChangerFvMesh.so)

target_link_libraries(${PROJECT_NAME} ${CANTERA_ROOT}/lib/libcantera_shared.so.2)

target_link_libraries(${PROJECT_NAME} dfCompressibleTurbulenceModels
dfChemistryModel
dfCanteraMixture
dfFluidThermophysicalModels
dfCombustionModels
${MPI_LIBRARIES}
$ENV{FOAM_LIBBIN}/openmpi-system/libPstream.so
)


install(TARGETS ${PROJECT_NAME} DESTINATION bin)