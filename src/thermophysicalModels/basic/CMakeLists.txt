project(dfFluidThermophysicalModels  LANGUAGES CXX)

# add library
set(workDir
    $ENV{SRC_ORIG}/thermophysicalModels/basic/)

set(SOURCES
    ${workDir}/basicThermo/basicThermo.C
    ${workDir}/fluidThermo/fluidThermo.C

    ${workDir}/psiThermo/psiThermo.C

    ${workDir}/rhoThermo/rhoThermo.C

    ${workDir}/derivedFvPatchFields/fixedEnergy/fixedEnergyFvPatchScalarField.C
    ${workDir}/derivedFvPatchFields/gradientEnergy/gradientEnergyFvPatchScalarField.C
    ${workDir}/derivedFvPatchFields/mixedEnergy/mixedEnergyFvPatchScalarField.C

    ${workDir}/derivedFvPatchFields/energyJump/energyJump/energyJumpFvPatchScalarField.C
    ${workDir}/derivedFvPatchFields/energyJump/energyJumpAMI/energyJumpAMIFvPatchScalarField.C
    )

add_library(${PROJECT_NAME} SHARED ${SOURCES})

include_directories(
    ${OpenFOAM_SRC}/transportModels/compressible/lnInclude
    ${OpenFOAM_SRC}/thermophysicalModels/basic/lnInclude
    ${OpenFOAM_SRC}/thermophysicalModels/thermophysicalProperties/lnInclude
    ${OpenFOAM_SRC}/meshTools/lnInclude
    $ENV{DF_SRC}/thermophysicalModels/thermophysicalProperties/solidProperties/solidMixtureProperties
    $ENV{DF_SRC}/thermophysicalModels/thermophysicalProperties/solidProperties/solidProperties
)



target_link_libraries(${PROJECT_NAME} $ENV{FOAM_LIBBIN}/libfiniteVolume.so libmeshTools.so libcompressibleTransportModels.so)

target_link_libraries(${PROJECT_NAME} ${CANTERA_ROOT}/lib/libcantera_shared.so.2)

target_link_libraries(${PROJECT_NAME} 

#${DF_ROOT}/lib/libdfThermophysicalProperties.so
dfThermophysicalProperties
)


# install
install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    )
