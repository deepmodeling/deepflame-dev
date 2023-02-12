project(dfTurbulenceModels LANGUAGES CXX)


include_directories(
${dfturbulence_inc}
    ${OpenFOAM_SRC}/TurbulenceModels/compressible/lnInclude 
    ${OpenFOAM_SRC}/TurbulenceModels/turbulenceModels/lnInclude 
    ${OpenFOAM_SRC}/transportModels/compressible/lnInclude 
    ${OpenFOAM_SRC}/thermophysicalModels/basic/lnInclude 
    ${OpenFOAM_SRC}/finiteVolume/lnInclude 
    ${OpenFOAM_SRC}/meshTools/lnInclude 
)

add_library(dfTurbulenceModels SHARED

${CMAKE_CURRENT_SOURCE_DIR}/makeTurbulentFluidThermoModels.C
)


# dynamic link
target_link_libraries(${PROJECT_NAME} $ENV{FOAM_LIBBIN}/libcompressibleTransportModels.so libturbulenceModels.so
libfiniteVolume.so libmeshTools.so )

target_link_libraries(${PROJECT_NAME} dfCompressibleTurbulenceModels
dfFluidThermophysicalModels
)




install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    )
