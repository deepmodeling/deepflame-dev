
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

#set the options for the shared library
set(LINK_FLAGS  "-fuse-ld=bfd -shared -Xlinker --add-needed -Xlinker --no-as-needed")

#set the link options
set(CMAKE_SHARED_LINKER_FLAGS "${LINK_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${LINK_FLAGS}")


SET(CMAKE_C_COMPILER g++)
set(PATH_LIB_OPENMPI "openmpi-system")  # Foundation version
set(DEFINITIONS_COMPILE "-std=c++11 -m64 -Dlinux64 -DWM_ARCH_OPTION=64 
-DWM_DP -DWM_LABEL_SIZE=32 -Wall -Wextra -Wold-style-cast -Wnon-virtual-dtor 
-Wno-unused-parameter -Wno-invalid-offsetof -Wno-attributes -O3  
-DNoRepository -ftemplate-depth-100 -std=c++14 
-Wno-unused-variable -Wno-unused-but-set-variable -Wno-old-style-cast -DOMPI_SKIP_MPICXX  
-pthread
-fPIC -c")


# Compiling configure
add_definitions("${DEFINITIONS_COMPILE}")


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
  find_package (Python REQUIRED COMPONENTS Interpreter Development)
  find_package(pybind11)
  include_directories(
  ${Python_INCLUDE_DIRS}
  ${pybind11_INCLUDE_DIR}/pybind11
  )
  #target_link_libraries(${PROJECT_NAME} ${Python_LIBRARIES})
  message(STATUS "pybind libraries" ${Python_LIBRARIES})
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
PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    )
