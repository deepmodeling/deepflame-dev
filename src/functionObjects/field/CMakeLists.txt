project(dfFieldFunctionObjects LANGUAGES CXX)

# Compiling configure
add_definitions("${SHARED_COMPILE_OPTION}")



# add library
SET(workDir ${SRC_ORIG}/functionObjects/field)

SET(SOURCES
${workDir}/fieldAverage/fieldAverage.C
${workDir}/fieldAverage/fieldAverageItem/fieldAverageItem.C
${workDir}/fieldAverage/fieldAverageItem/fieldAverageItemIO.C

${workDir}/fieldCoordinateSystemTransform/fieldCoordinateSystemTransform.C
${workDir}/fieldMinMax/fieldMinMax.C

${workDir}/fieldValues/fieldValue/fieldValue.C
${workDir}/fieldValues/fieldValue/fieldValueNew.C
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
add_library(dfFieldFunctionObjects SHARED ${SOURCES})

# dynamic link
target_link_libraries(${PROJECT_NAME} ${OpenFOAM_LIB_DIR}/libfiniteVolume.so
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