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

    $ENV{CANTERA_ROOT}/include
    #$ENV{WM_DIR}/rules/mplibType 
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
)

target_link_libraries(${PROJECT_NAME} ${CANTERA_ROOT}/lib/libcantera_shared.so.2)



install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    )
