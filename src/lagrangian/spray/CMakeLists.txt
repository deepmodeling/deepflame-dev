project(dfLagrangianSpray
LANGUAGES CXX)
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
PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    )
