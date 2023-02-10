project(dfCombustionModels LANGUAGES CXX)

#set(INC_DIR R_SEARCH(search_paths))

message(STATUS "header" ${header_dir})

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
)

target_link_libraries(${PROJECT_NAME} ${CANTERA_ROOT}/lib/libcantera_shared.so.2)



install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    )
