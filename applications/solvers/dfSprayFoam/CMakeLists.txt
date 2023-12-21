cmake_minimum_required(VERSION 3.5)
project(dfSprayFoam LANGUAGES CXX)
FIND_PACKAGE(MPI REQUIRED)

# Check valid thirdParty
if(DEFINED ENV{WM_PROJECT_DIR})
	MESSAGE(STATUS "OpenFOAM: " $ENV{WM_PROJECT_DIR})
else()
	message(FATAL_ERROR "OpenFOAM is not sourced")
endif(DEFINED ENV{WM_PROJECT_DIR})

if(DEFINED ENV{CANTERA_ROOT})
	MESSAGE(STATUS "libcantera: " $ENV{CANTERA_ROOT})
  SET(CANTERA_ROOT $ENV{CANTERA_ROOT})
else()
	message(FATAL_ERROR "libcantera directory is not specified")
endif(DEFINED ENV{CANTERA_ROOT})

# define variables
SET(OpenFOAM_LIB_DIR $ENV{FOAM_LIBBIN})
SET(OpenFOAM_SRC $ENV{FOAM_SRC})

SET(DF_ROOT $ENV{DF_ROOT})
SET(DF_SRC $ENV{DF_SRC})
SET(SRC_ORIG $ENV{SRC_ORIG})

# set compilation options
SET(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=bfd -Xlinker --add-needed -Xlinker --no-as-needed")

SET(CMAKE_C_COMPILER g++)
SET(PATH_LIB_OPENMPI "openmpi-system")  # Foundation version
SET(EXE_COMPILE_OPTION "-std=c++14 -m64 -Dlinux64 -DWM_ARCH_OPTION=64
-DWM_DP -DWM_LABEL_SIZE=32 -Wall -Wextra -Wold-style-cast -Wnon-virtual-dtor
-Wno-unused-parameter -Wno-invalid-offsetof -Wno-attributes -O3
-DNoRepository -ftemplate-depth-100
-Wno-unused-variable -Wno-unused-but-set-variable -Wno-old-style-cast -DOMPI_SKIP_MPICXX
-pthread -fPIC")
add_definitions("${EXE_COMPILE_OPTION}")

# add header files
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
R_SEARCH(${DF_SRC}/thermophysicalModels/basic dfthermophysicalbasic_inc)
R_SEARCH(${DF_SRC}/thermophysicalModels/SLGThermo dfthermophysicalslg_inc)
R_SEARCH(${DF_SRC}/TurbulenceModels dfturbulence_inc)
R_SEARCH(${DF_SRC}/dynamicMesh dfnewdynamic_inc)
R_SEARCH(${DF_SRC}/dynamicFvMesh dffvdynamic_inc)

include_directories(
    ${OpenFOAM_SRC}/finiteVolume/lnInclude
    ${OpenFOAM_SRC}/OSspecific/POSIX/lnInclude
    ${OpenFOAM_SRC}/OpenFOAM/lnInclude
    ${OpenFOAM_SRC}/finiteVolume/lnInclude
    ${OpenFOAM_SRC}/meshTools/lnInclude
    ${OpenFOAM_SRC}/sampling/lnInclude
    ${OpenFOAM_SRC}/TurbulenceModels/turbulenceModels/lnInclude
    ${OpenFOAM_SRC}/TurbulenceModels/compressible/lnInclude
    ${OpenFOAM_SRC}/regionModels/regionModel/lnInclude
    ${OpenFOAM_SRC}/regionModels/surfaceFilmModels/lnInclude
    ${OpenFOAM_SRC}/dynamicFvMesh/lnInclude
    ${OpenFOAM_SRC}/meshTools/lnInclude
    ${OpenFOAM_SRC}/lagrangian/basic/lnInclude
    ${dflagrangianinter_inc}
    ${OpenFOAM_SRC}/lagrangian/intermediate/lnInclude
    ${dflagrangianspray_inc}
    ${OpenFOAM_SRC}/lagrangian/spray/lnInclude
    ${OpenFOAM_SRC}/lagrangian/distributionModels/lnInclude
    ${OpenFOAM_SRC}/transportModels/compressible/lnInclude
    ${OpenFOAM_SRC}/thermophysicalModels/basic/lnInclude
    ${dfthermophysicalprop_inc}
    ${OpenFOAM_SRC}/thermophysicalModels/thermophysicalProperties/lnInclude
    ${dfthermophysicalslg_inc}
    ${dfcantera_inc}
    ${dfchemistry_inc}
    ${dfcombustion_inc}
    ${OpenFOAM_SRC}/Pstream/mpi
    ${CANTERA_ROOT}/include
    ${MPI_INCLUDE_PATH}
    ${PROJECT_SOURCE_DIR}
)

# add execution
add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/dfSprayFoam.C)

target_link_libraries(${PROJECT_NAME}
    $ENV{FOAM_LIBBIN}/libfiniteVolume.so libmeshTools.so libcompressibleTransportModels.so libturbulenceModels.so
    libsampling.so libOpenFOAM.so libdynamicFvMesh.so liblagrangian.so libregionModels.so
    ${CANTERA_ROOT}/lib/libcantera_shared.so.2
    ${DF_ROOT}/lib/libdfChemistryModel.so
    ${DF_ROOT}/lib/libdfCanteraMixture.so
    ${DF_ROOT}/lib/libdfFluidThermophysicalModels.so
    ${DF_ROOT}/lib/libdfCombustionModels.so
    ${DF_ROOT}/lib/libdfLagrangianIntermediate.so
    ${DF_ROOT}/lib/libdfLagrangianTurbulence.so
    ${DF_ROOT}/lib/libdfLagrangianSpray.so
    ${DF_ROOT}/lib/libdfThermophysicalProperties.so
    ${DF_ROOT}/lib/libdfCompressibleTurbulenceModels.so
    ${DF_ROOT}/lib/libdfSLGThermo.so
    ${DF_ROOT}/lib/libdfSurfaceFilmModels.so
)

if(DEFINED ENV{PYTHON_INC_DIR})
  add_definitions(-DUSE_PYTORCH)
  # https://pybind11.readthedocs.io/en/stable/advanced/embedding.html
  find_package(pybind11)
  target_link_libraries(${PROJECT_NAME} pybind11::embed)
endif()

# install
set(CMAKE_INSTALL_PREFIX ${DF_ROOT})
install(TARGETS ${PROJECT_NAME} DESTINATION bin)
