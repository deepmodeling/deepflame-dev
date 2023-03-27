cmake_minimum_required(VERSION 3.5)
project(deepflame LANGUAGES CXX)
FIND_PACKAGE(MPI REQUIRED)
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

set(SRC_ORIG ${CMAKE_CURRENT_SOURCE_DIR}/src_orig)


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

set(DF_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(DF_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src)
       
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

add_subdirectory(${DF_SRC}/thermophysicalModels/thermophysicalProperties)
add_subdirectory(${DF_SRC}/thermophysicalModels/basic)
add_subdirectory(${DF_SRC}/functionObjects/field)
add_subdirectory(${DF_SRC}/dfCanteraMixture)
add_subdirectory(${DF_SRC}/thermophysicalModels/SLGThermo)
add_subdirectory(${DF_SRC}/dfChemistryModel)
add_subdirectory(${DF_SRC}/TurbulenceModels/compressible)
add_subdirectory(${DF_SRC}/TurbulenceModels/turbulenceModels)
add_subdirectory(${DF_SRC}/regionModels/surfaceFilmModels)
add_subdirectory(${DF_SRC}/lagrangian/turbulence)
add_subdirectory(${DF_SRC}/lagrangian/spray)
add_subdirectory(${DF_SRC}/lagrangian/intermediate)
add_subdirectory(${DF_SRC}/dfCombustionModels)
add_subdirectory(${DF_SRC}/dynamicMesh)
add_subdirectory(${DF_SRC}/dynamicFvMesh)


# add_subdirectory(${DF_ROOT}/applications/solvers/df0DFoam)
#add_subdirectory(${DF_ROOT}/applications/solvers/dfLowMachFoam)
# add_subdirectory(${DF_ROOT}/applications/solvers/dfHighSpeedFoam)
# add_subdirectory(${DF_ROOT}/applications/solvers/dfSprayFoam)
