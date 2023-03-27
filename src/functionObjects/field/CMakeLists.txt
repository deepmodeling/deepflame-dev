project(dfFieldFunctionObjects LANGUAGES CXX)

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
set(workDir ${SRC_ORIG}/functionObjects/field)


# add library
add_library(dfFieldFunctionObjects SHARED

${workDir}/fieldAverage/fieldAverage.C
${workDir}/fieldAverage/fieldAverageItem/fieldAverageItem.C
${workDir}/fieldAverage/fieldAverageItem/fieldAverageItemIO.C

${workDir}/fieldCoordinateSystemTransform/fieldCoordinateSystemTransform.C
${workDir}/fieldMinMax/fieldMinMax.C

${workDir}/fieldValues/fieldValue/fieldValue.C
${workDir}/fieldValues/fieldValue/fieldValueNew.C
${workDir}/fieldValues/fieldValueDelta/fieldValueDelta.C
${workDir}/fieldValues/volFieldValue/volFieldValue.C
${workDir}/fieldValues/surfaceFieldValue/surfaceFieldValue.C

${workDir}/nearWallFields/nearWallFields.C
${workDir}/nearWallFields/findCellParticle.C
${workDir}/nearWallFields/findCellParticleCloud.C

${workDir}/processorField/processorField.C
${workDir}/readFields/readFields.C

${workDir}/streamLine/streamLine.C
${workDir}/streamLine/streamLineParticle.C
${workDir}/streamLine/streamLineParticleCloud.C

${workDir}/surfaceInterpolate/surfaceInterpolate.C

${workDir}/regionSizeDistribution/regionSizeDistribution.C
${workDir}/histogram/histogram.C

${workDir}/fieldExpression/fieldExpression.C
${workDir}/components/components.C
${workDir}/randomise/randomise.C
${workDir}/div/div.C
${workDir}/grad/grad.C
${workDir}/ddt/ddt.C
${workDir}/mag/mag.C
${workDir}/magSqr/magSqr.C
${workDir}/scale/scale.C
${workDir}/log/log.C
${workDir}/vorticity/vorticity.C
${workDir}/enstrophy/enstrophy.C
${workDir}/Q/Q.C
${workDir}/Lambda2/Lambda2.C
${workDir}/flowType/flowType.C
${workDir}/CourantNo/CourantNo.C

${workDir}/blendingFactor/blendingFactor.C
${workDir}/pressure/pressure.C
${workDir}/MachNo/MachNo.C

${workDir}/yPlus/yPlus.C
${workDir}/turbulenceIntensity/turbulenceIntensity.C

${workDir}/writeCellCentres/writeCellCentres.C
${workDir}/writeCellVolumes/writeCellVolumes.C

${workDir}/XiReactionRate/XiReactionRate.C
${workDir}/streamFunction/streamFunction.C

${workDir}/fieldsExpression/fieldsExpression.C
${workDir}/add/add.C
${workDir}/subtract/subtract.C

${workDir}/interfaceHeight/interfaceHeight.C

${workDir}/age/age.C
)

# dynamic link
target_link_libraries(${PROJECT_NAME} $ENV{FOAM_LIBBIN}/libfiniteVolume.so 
libmeshTools.so 
libincompressibleTransportModels.so 
libturbulenceModels.so 
libcompressibleTransportModels.so
libsurfMesh.so 
liblagrangian.so 
libfileFormats.so 
libsampling.so
libsurfMesh.so
)
target_link_libraries(${PROJECT_NAME} dfFluidThermophysicalModels)




include_directories(
    ${SRC_ORIG}/functionObjects/field/lnInclude
    ${OpenFOAM_SRC}/meshTools/lnInclude 
    ${OpenFOAM_SRC}/lagrangian/basic/lnInclude
    ${OpenFOAM_SRC}/fileFormats/lnInclude 
    ${OpenFOAM_SRC}/surfMesh/lnInclude 
    ${OpenFOAM_SRC}/sampling/lnInclude 
    ${OpenFOAM_SRC}/thermophysicalModels/basic/lnInclude
    ${OpenFOAM_SRC}/transportModels/compressible/lnInclude 
    ${OpenFOAM_SRC}/transportModels
    ${OpenFOAM_SRC}/TurbulenceModels/turbulenceModels/lnInclude
    
    )
    


# install
install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    )