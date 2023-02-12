
project(dfCompressibleTurbulenceModels LANGUAGES CXX)

set(WORKINGDIR ${SRC_ORIG}/TurbulenceModels/compressible/)
include_directories(
                    ${OpenFOAM_SRC}/TurbulenceModels/compressible/lnInclude
                    ${OpenFOAM_SRC}/TurbulenceModels/turbulenceModels/lnInclude
                    ${OpenFOAM_SRC}/transportModels/compressible/lnInclude
                    ${OpenFOAM_SRC}/thermophysicalModels/basic/lnInclude
                    ${OpenFOAM_SRC}/meshTools/lnInclude
                    ${CANTERA_ROOT}/include
                    )




add_library(dfCompressibleTurbulenceModels SHARED


${WORKINGDIR}/compressibleTurbulenceModel.C
${WORKINGDIR}/turbulentFluidThermoModels/turbulentFluidThermoModels.C
${WORKINGDIR}/turbulentFluidThermoModels/derivedFvPatchFields/wallFunctions/alphatWallFunctions/alphatWallFunction/alphatWallFunctionFvPatchScalarField.C
)


# dynamic link
target_link_libraries(${PROJECT_NAME} $ENV{FOAM_LIBBIN}/libcompressibleTransportModels.so libturbulenceModels.so libfiniteVolume.so libmeshTools.so)





install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    )
