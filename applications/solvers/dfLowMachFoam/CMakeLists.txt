cmake_minimum_required(VERSION 3.5)
project(dfLowMachFoam LANGUAGES CXX)
FIND_PACKAGE(MPI REQUIRED)

message(STATUS "MPI include path is " ${MPI_INCLUDE_PATH})
message(STATUS "MPI libraries " ${MPI_LIBRARIES})


# Check valid OpenFOAM
if(DEFINED ENV{WM_PROJECT_DIR})
	MESSAGE(STATUS "OpenFOAM: " $ENV{WM_PROJECT_DIR})
else()
	message(FATAL_ERROR "OpenFOAM is not sourced")
endif(DEFINED ENV{WM_PROJECT_DIR})
set(OpenFOAM_VERSION $ENV{WM_PROJECT_VERSION}) 
set(OpenFOAM_DIR $ENV{WM_PROJECT_DIR})
set(OpenFOAM_LIB_DIR $ENV{FOAM_LIBBIN})
set(OpenFOAM_SRC $ENV{FOAM_SRC})

set(DF_ROOT $ENV{DF_ROOT})
set(DF_SRC ${DF_ROOT}/src)
set(SRC_ORIG ${DF_ROOT}/src_orig)


FUNCTION(R_SEARCH search_path return_list)
    FILE(GLOB_RECURSE new_list ${search_path}/*.H)
    SET(dir_list "")
    FOREACH(file_path ${new_list})
        GET_FILENAME_COMPONENT(dir_path ${file_path} PATH)
        SET(dir_list ${dir_list} ${dir_path})
    ENDFOREACH()
    LIST(REMOVE_DUPLICATES dir_list)
    SET(${return_list} ${dir_list} PARENT_SCOPE)
ENDFUNCTION(R_SEARCH)


       

R_SEARCH(${DF_SRC}/dfCombustionModels dfcombustion_inc)
R_SEARCH(${DF_SRC}/dfCanteraMixture dfcantera_inc)
R_SEARCH(${DF_SRC}/lagrangian/intermediate dflagrangianinter_inc)
R_SEARCH(${DF_SRC}/lagrangian/spray dflagrangianspray_inc)
R_SEARCH(${DF_SRC}/lagrangian/turbulence dflagrangianturb_inc)
R_SEARCH(${DF_SRC}/dfChemistryModel dfchemistry_inc)
R_SEARCH(${DF_SRC}/thermophysicalModels/thermophysicalProperties dfthermophysicalprop_inc)
R_SEARCH(${DF_SRC}/thermophysicalModels/thermophysicalProperties dfthermophysicalprop_inc)
R_SEARCH(${DF_SRC}/thermophysicalModels/basic dfthermophysicalbasic_inc)
R_SEARCH(${DF_SRC}/thermophysicalModels/SLGThermo dfthermophysicalslg_inc)
R_SEARCH(${DF_SRC}/TurbulenceModels dfturbulence_inc)
R_SEARCH(${DF_SRC}/dynamicMesh dfnewdynamic_inc)
R_SEARCH(${DF_SRC}/dynamicFvMesh dffvdynamic_inc)


# Check valid libcantera
if(DEFINED ENV{CANTERA_ROOT})
	MESSAGE(STATUS "libcantera: " $ENV{CANTERA_ROOT})
  set(CANTERA_ROOT $ENV{CANTERA_ROOT})
else()
	message(FATAL_ERROR "libcantera directory is not specified")
endif(DEFINED ENV{CANTERA_ROOT}) 


# Check valid Torch
#if(DEFINED $ENV{PYTHON_INC_DIR)})
#	MESSAGE(STATUS "Torch: " $ENV{CANTERA_ROOT})
#else()
#	message(FATAL_ERROR "no torch used")
#endif(DEFINED ENV{PYTHON_INC_DIR}) 



include_directories(
                    ${OpenFOAM_SRC}/finiteVolume/lnInclude
                   
                    ${OpenFOAM_SRC}/OSspecific/POSIX/lnInclude
                    ${OpenFOAM_SRC}/OpenFOAM/lnInclude
                    )

message(STATUS "dummydir:" ${OpenFOAM_LIB_DIR}/dummy)
link_directories(${OpenFOAM_LIB_DIR} ${OpenFOAM_LIB_DIR}/dummy )

set(CMAKE_INSTALL_PREFIX ${DF_ROOT})

#set the options for the shared library
set(LINK_FLAGS  "-fuse-ld=bfd -Xlinker --add-needed -Xlinker --no-as-needed")

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
-pthread -fPIC")


# Compiling configure
add_definitions("${DEFINITIONS_COMPILE}")

FIND_PACKAGE(MPI REQUIRED)

include_directories(
    ${OpenFOAM_SRC}/transportModels/compressible/lnInclude 
    ${OpenFOAM_SRC}/thermophysicalModels/basic/lnInclude 
    ${OpenFOAM_SRC}/TurbulenceModels/turbulenceModels/lnInclude 
    ${OpenFOAM_SRC}/TurbulenceModels/compressible/lnInclude 
    ${OpenFOAM_SRC}/finiteVolume/cfdTools 
    ${OpenFOAM_SRC}/finiteVolume/lnInclude 
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


add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/dfLowMachFoam.C)

if(DEFINED ENV{PYTHON_INC_DIR})
  add_definitions(-DUSE_PYTORCH)
  find_package (Python REQUIRED COMPONENTS Interpreter Development)
  find_package(pybind11)
  include_directories(
  ${Python_INCLUDE_DIRS}
  ${pybind11_INCLUDE_DIR}/pybind11
  )
  target_link_libraries(${PROJECT_NAME} ${Python_LIBRARIES})
endif()

target_link_libraries(${PROJECT_NAME} $ENV{FOAM_LIBBIN}/libfiniteVolume.so libmeshTools.so libcompressibleTransportModels.so libturbulenceModels.so libsampling.so libOpenFOAM.so )

target_link_libraries(${PROJECT_NAME} ${CANTERA_ROOT}/lib/libcantera_shared.so.2)

target_link_libraries(${PROJECT_NAME} /home/runze/JX/deepflame-cmake/lib/libdfCanteraMixture.so
/home/runze/JX/deepflame-cmake/lib/libdfChemistryModel.so
/home/runze/JX/deepflame-cmake/lib/libdfCanteraMixture.so
/home/runze/JX/deepflame-cmake/lib/libdfFluidThermophysicalModels.so
/home/runze/JX/deepflame-cmake/lib/libdfCombustionModels.so
)
link_directories(${OpenFOAM_LIB_DIR} ${OpenFOAM_LIB_DIR}/dummy)


target_link_libraries(${PROJECT_NAME} 
$ENV{FOAM_LIBBIN}/openmpi-system/libPstream.so
${MPI_LIBRARIES}
)


#set_target_properties(${PROJECT_NAME}
#     PROPERTIES
#     LINK_OPTIONS "-Xlinker --add-needed -Xlinker --no-as-needed"
#     )


install(TARGETS ${PROJECT_NAME} DESTINATION bin)