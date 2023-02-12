project(dfLagrangianSpray
LANGUAGES CXX)

include_directories(
    ${OpenFOAM_SRC}/lagrangian/basic/lnInclude 
    ${dflagrangianinter_inc} 
    ${OpenFOAM_SRC}/lagrangian/intermediate/lnInclude 
    ${OpenFOAM_SRC}/lagrangian/distributionModels/lnInclude 
    ${dflagrangianspray_inc} 
    ${OpenFOAM_SRC}/lagrangian/spray/lnInclude 
    ${dflagrangianturb_inc} 
    ${OpenFOAM_SRC}/lagrangian/turbulence/lnInclude 
    ${OpenFOAM_SRC}/transportModels/compressible/lnInclude 
    ${OpenFOAM_SRC}/thermophysicalModels/basic/lnInclude 
    ${dfthermophysicalprop_inc} 
    ${OpenFOAM_SRC}/thermophysicalModels/thermophysicalProperties/lnInclude 
    ${dfthermophysicalslg_inc} 
    $ENV{CANTERA_ROOT}/include
    ${dfcantera_inc} 
    ${OpenFOAM_SRC}/TurbulenceModels/turbulenceModels/lnInclude 
    ${OpenFOAM_SRC}/TurbulenceModels/incompressible/lnInclude 
    ${OpenFOAM_SRC}/TurbulenceModels/compressible/lnInclude 
    ${OpenFOAM_SRC}/transportModels 
    ${OpenFOAM_SRC}/regionModels/regionModel/lnInclude 
    ${OpenFOAM_SRC}/regionModels/surfaceFilmModels/lnInclude 
    ${OpenFOAM_SRC}/dynamicFvMesh/lnInclude 
    ${OpenFOAM_SRC}/sampling/lnInclude 
    ${OpenFOAM_SRC}/finiteVolume/lnInclude 
    ${OpenFOAM_SRC}/meshTools/lnInclude

)


set(SPRAYPARCEL ${CMAKE_CURRENT_SOURCE_DIR}/parcels/derived/basicSprayParcel)


add_library(dfLagrangianSpray SHARED

${CMAKE_CURRENT_SOURCE_DIR}/clouds/baseClasses/sprayCloud/sprayCloud.C

${SPRAYPARCEL}/defineBasicSprayParcel.C
${SPRAYPARCEL}/makeBasicSprayParcelSubmodels.C
)


# dynamic link
target_link_libraries(${PROJECT_NAME} $ENV{FOAM_LIBBIN}/libcompressibleTransportModels.so libturbulenceModels.so
libfiniteVolume.so libmeshTools.so liblagrangian.so libdistributionModels.so libincompressibleTransportModels.so libdynamicFvMesh.so libsampling.so
)

target_link_libraries(${PROJECT_NAME} dfCompressibleTurbulenceModels
dfCanteraMixture
dfLagrangianIntermediate
dfFluidThermophysicalModels
dfThermophysicalProperties
dfSLGThermo
dfLagrangianTurbulence
)

target_link_libraries(${PROJECT_NAME} ${CANTERA_ROOT}/lib/libcantera_shared.so.2)



install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    )
