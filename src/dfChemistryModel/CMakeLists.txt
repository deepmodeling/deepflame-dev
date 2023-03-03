project(dfChemistryModel LANGUAGES CXX)

set(CMAKE_PREFIX_PATH $ENV{LIBTORCH_ROOT})

find_package(MPI REQUIRED)
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

include_directories(
    ${dfchemistry_inc}
    ${OpenFOAM_SRC}/meshTools/lnInclude
    ${OpenFOAM_SRC}/thermophysicalModels/basic/lnInclude
    ${OpenFOAM_SRC}/transportModels/compressible/lnInclude
    ${OpenFOAM_SRC}/TurbulenceModels/turbulenceModels/lnInclude
    ${OpenFOAM_SRC}/TurbulenceModels/compressible/lnInclude
    ${dfcantera_inc}
    ${CANTERA_ROOT}/include
    ${OpenFOAM_SRC}/Pstream/mpi
    ${MPI_INCLUDE_PATH}
)

add_library(dfChemistryModel SHARED

${DF_SRC}/${PROJECT_NAME}/loadBalancing/ChemistryProblem.C
${DF_SRC}/${PROJECT_NAME}/loadBalancing/ChemistrySolution.C
${DF_SRC}/${PROJECT_NAME}/loadBalancing/ChemistryLoad.C
${DF_SRC}/${PROJECT_NAME}/loadBalancing/LoadBalancerBase.C
${DF_SRC}/${PROJECT_NAME}/loadBalancing/SendBuffer.C
${DF_SRC}/${PROJECT_NAME}/loadBalancing/RecvBuffer.C
${DF_SRC}/${PROJECT_NAME}/loadBalancing/algorithms_DLB.C
${DF_SRC}/${PROJECT_NAME}/loadBalancing/runtime_assert.C
${DF_SRC}/${PROJECT_NAME}/loadBalancing/LoadBalancer.C

${DF_SRC}/${PROJECT_NAME}/makeDfChemistryModels.C
)

# if(DEFINED ENV{PYTHON_INC_DIR})
#   add_definitions(-DUSE_PYTORCH)
#   include_directories(
#   /usr/include/python3.8
#   /home/runze/.local/lib/python3.8/site-packages/pybind11/include
#   )
# endif()
if(DEFINED ENV{LIBTORCH_ROOT})
  add_definitions(-DUSE_LIBTORCH)
  include_directories(
  ${TORCH_INCLUDE_PATH}
  )
  target_link_libraries(
  ${PROJECT_NAME} ${TORCH_LIBRARIES}
  )
endif()

message(${CUDA_INCLUDE_DIRS})
message(${CUDA_LIBRARIES})


# dynamic link
target_link_libraries(${PROJECT_NAME} $ENV{FOAM_LIBBIN}/libcompressibleTransportModels.so libturbulenceModels.so)

target_link_libraries(${PROJECT_NAME} ${CANTERA_ROOT}/lib/libcantera_shared.so.2)

target_link_libraries(${PROJECT_NAME} dfFluidThermophysicalModels
dfCompressibleTurbulenceModels
dfCanteraMixture
${MPI_LIBRARIES}
)


install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    )