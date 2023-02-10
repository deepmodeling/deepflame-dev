
project(dfChemistryModel LANGUAGES CXX)

include_directories(
                    ${dfchemistry_inc}
                    ${OpenFOAM_SRC}/meshTools/lnInclude
                    ${OpenFOAM_SRC}/thermophysicalModels/basic/lnInclude
                    ${OpenFOAM_SRC}/transportModels/compressible/lnInclude
                    ${OpenFOAM_SRC}/TurbulenceModels/turbulenceModels/lnInclude
                    ${OpenFOAM_SRC}/TurbulenceModels/compressible/lnInclude
                    ${dfcantera_inc}
                    ${CANTERA_ROOT}/include
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

if(DEFINED ENV{PYTHON_INC_DIR})
  add_definitions(-DUSE_PYTORCH)
  include_directories(
  /usr/include/python3.8
  /home/runze/.local/lib/python3.8/site-packages/pybind11/include
  )
endif()






# dynamic link
target_link_libraries(${PROJECT_NAME} $ENV{FOAM_LIBBIN}/libcompressibleTransportModels.so libturbulenceModels.so)

target_link_libraries(${PROJECT_NAME} ${CANTERA_ROOT}/lib/libcantera_shared.so.2)

target_link_libraries(${PROJECT_NAME} dfFluidThermophysicalModels
dfCompressibleTurbulenceModels
dfCanteraMixture
)


install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    )
