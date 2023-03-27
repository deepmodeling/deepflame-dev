project(dfCombustionModels LANGUAGES CXX)

#set(INC_DIR R_SEARCH(search_paths))

message(STATUS "header" ${header_dir})

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

include_directories(

    ${OpenFOAM_SRC}/transportModels/compressible/lnInclude 
    ${OpenFOAM_SRC}/thermophysicalModels/basic/lnInclude 
    ${OpenFOAM_SRC}/TurbulenceModels/turbulenceModels/lnInclude 
    ${OpenFOAM_SRC}/TurbulenceModels/compressible/lnInclude 
    ${OpenFOAM_SRC}/finiteVolume/lnInclude 
    ${OpenFOAM_SRC}/meshTools/lnInclude 
    ${DF_SRC}/dfCanteraMixture
    ${PROJECT_SOURCE_DIR}
    ${dfcombustion_inc}
    ${dfchemistry_inc}

    ${OpenFOAM_SRC}/Pstream/mpi 
    ${OpenFOAM_SRC}/ODE/lnInclude
    ${MPI_INCLUDE_PATH}
 


    $ENV{CANTERA_ROOT}/include
    /usr/include/python3.8
    /home/runze/.local/lib/python3.8/site-packages/pybind11/include 
    #$ENV{WM_DIR}/rules/mplibType 
)

#if(DEFINED ENV{LIBTORCH_ROOT})
#  add_definitions(-DUSE_LIBTORCH)
#  include_directories($ENV{LIBTORCH_ROOT}/include
#  $ENV{LIBTORCH_ROOT}/include/torch/csrc/api/include
#  )
  
#  target_link_libraries(
#  $(LIBTORCH_ROOT)/lib/libtorch.so
#  $(LIBTORCH_ROOT)/lib/libc10.so
  #-rdynamic
  #-lpthread
#  )
if(DEFINED ENV{PYTHON_INC_DIR})
  add_definitions(-DUSE_PYTORCH)
  include_directories(
  /usr/include/python3.8
  /home/runze/.local/lib/python3.8/site-packages/pybind11/include
  )
else()
endif()

add_library(dfCombustionModels SHARED

${DF_SRC}/${PROJECT_NAME}/combustionModel/combustionModel.C
${DF_SRC}/${PROJECT_NAME}/CombustionModel/CombustionModel/CombustionModels.C
${DF_SRC}/${PROJECT_NAME}/PaSR/PaSRs.C
${DF_SRC}/${PROJECT_NAME}/EDC/EDCs.C

${DF_SRC}/${PROJECT_NAME}/laminar/laminars.C

${DF_SRC}/${PROJECT_NAME}/noCombustion/noCombustions.C
)


# dynamic link
target_link_libraries(${PROJECT_NAME} $ENV{FOAM_LIBBIN}/libcompressibleTransportModels.so libturbulenceModels.so
libfiniteVolume.so libmeshTools.so )

target_link_libraries(${PROJECT_NAME} dfCompressibleTurbulenceModels
dfCanteraMixture
dfChemistryModel
${MPI_LIBRARIES}
$ENV{FOAM_LIBBIN}/openmpi-system/libPstream.so

)

target_link_libraries(${PROJECT_NAME} ${CANTERA_ROOT}/lib/libcantera_shared.so.2)



install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    )
