project(dfChemistryModel LANGUAGES CXX)

FIND_PACKAGE(MPI REQUIRED)

# Compiling configure
add_definitions("${SHARED_COMPILE_OPTION}")

include_directories(
                    ${dfchemistry_inc}
                    ${OpenFOAM_SRC}/meshTools/lnInclude
                    ${OpenFOAM_SRC}/thermophysicalModels/basic/lnInclude
                    ${OpenFOAM_SRC}/transportModels/compressible/lnInclude
                    ${OpenFOAM_SRC}/TurbulenceModels/turbulenceModels/lnInclude
                    ${OpenFOAM_SRC}/TurbulenceModels/compressible/lnInclude
                    ${dfcantera_inc}
                    ${MPI_INCLUDE_PATH}
                    ${OpenFOAM_SRC}/Pstream/mpi
                    ${CANTERA_ROOT}/include
                    )


SET(workDir ${DF_SRC}/${PROJECT_NAME})
SET(LB ${workDir}/loadBalancing)
SET(SOURCES 

${LB}/ChemistryProblem.C
${LB}/ChemistrySolution.C
${LB}/ChemistryLoad.C
${LB}/LoadBalancerBase.C
${LB}/SendBuffer.C
${LB}/RecvBuffer.C
${LB}/algorithms_DLB.C
${LB}/runtime_assert.C
${LB}/LoadBalancer.C
${workDir}/makeDfChemistryModels.C
)
add_library(dfChemistryModel SHARED ${SOURCES})

if(DEFINED ENV{PYTHON_INC_DIR})
  add_definitions(-DUSE_PYTORCH)
  find_package (Python REQUIRED COMPONENTS Interpreter Development)
  find_package(pybind11)
  include_directories(
  ${Python_INCLUDE_DIRS}
  ${pybind11_INCLUDE_DIR}/pybind11
  )
endif()






# dynamic link
target_link_libraries(${PROJECT_NAME} ${OpenFOAM_LIB_DIR}/libcompressibleTransportModels.so libturbulenceModels.so)

target_link_libraries(${PROJECT_NAME} ${CANTERA_ROOT}/lib/libcantera.so.2)

target_link_libraries(${PROJECT_NAME} dfFluidThermophysicalModels
dfCompressibleTurbulenceModels
dfCanteraMixture
${MPI_LIBRARIES}
)


install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    )
