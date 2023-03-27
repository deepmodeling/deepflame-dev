project(dfSurfaceFilmModels LANGUAGES CXX)

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
set(workDir ${SRC_ORIG}/regionModels/surfaceFilmModels)
set(KINEMATICMODELS ${workDir}/submodels/kinematic)
set(THERMOMODELS ${workDir}/submodels/thermo)
set(PATCHFIELDS ${workDir}/derivedFvPatchFields)

# add library
add_library (dfSurfaceFilmModels SHARED

#* Surface film models */
${workDir}/surfaceFilmModel/surfaceFilmModel.C
${workDir}/surfaceFilmModel/surfaceFilmModelNew.C
${workDir}/surfaceFilmRegionModel/surfaceFilmRegionModel.C
${workDir}/noFilm/noFilm.C
${workDir}/kinematicSingleLayer/kinematicSingleLayer.C
${workDir}/thermoSingleLayer/thermoSingleLayer.C


#Sub-models 
${workDir}/submodels/filmSubModelBase.C

${KINEMATICMODELS}/force/force/force.C
${KINEMATICMODELS}/force/force/forceNew.C
${KINEMATICMODELS}/force/forceList/forceList.C
${KINEMATICMODELS}/force/contactAngleForces/contactAngleForce/contactAngleForce.C
${KINEMATICMODELS}/force/contactAngleForces/distribution/distributionContactAngleForce.C
${KINEMATICMODELS}/force/contactAngleForces/temperatureDependent/temperatureDependentContactAngleForce.C
${KINEMATICMODELS}/force/contactAngleForces/perturbedTemperatureDependent/perturbedTemperatureDependentContactAngleForce.C
${KINEMATICMODELS}/force/thermocapillaryForce/thermocapillaryForce.C

${KINEMATICMODELS}/injectionModel/injectionModel/injectionModel.C
${KINEMATICMODELS}/injectionModel/injectionModel/injectionModelNew.C
${KINEMATICMODELS}/injectionModel/injectionModelList/injectionModelList.C
${KINEMATICMODELS}/injectionModel/drippingInjection/drippingInjection.C
${KINEMATICMODELS}/injectionModel/BrunDrippingInjection/BrunDrippingInjection.C
${KINEMATICMODELS}/injectionModel/patchInjection/patchInjection.C
${KINEMATICMODELS}/injectionModel/curvatureSeparation/curvatureSeparation.C

${KINEMATICMODELS}/transferModels/transferModel/transferModel.C
${KINEMATICMODELS}/transferModels/transferModel/transferModelNew.C
${KINEMATICMODELS}/transferModels/transferModelList/transferModelList.C

${KINEMATICMODELS}/filmThermoModel/filmThermoModel/filmThermoModel.C
${KINEMATICMODELS}/filmThermoModel/filmThermoModel/filmThermoModelNew.C
${KINEMATICMODELS}/filmThermoModel/constantFilmThermo/constantFilmThermo.C
${KINEMATICMODELS}/filmThermoModel/liquidFilmThermo/liquidFilmThermo.C

${KINEMATICMODELS}/filmTurbulenceModel/filmTurbulenceModel/filmTurbulenceModel.C
${KINEMATICMODELS}/filmTurbulenceModel/filmTurbulenceModel/filmTurbulenceModelNew.C
${KINEMATICMODELS}/filmTurbulenceModel/laminar/laminar.C

${THERMOMODELS}/phaseChangeModel/phaseChangeModel/phaseChangeModel.C
${THERMOMODELS}/phaseChangeModel/phaseChangeModel/phaseChangeModelNew.C
${THERMOMODELS}/phaseChangeModel/noPhaseChange/noPhaseChange.C
${THERMOMODELS}/phaseChangeModel/standardPhaseChange/standardPhaseChange.C
${THERMOMODELS}/phaseChangeModel/solidification/solidification.C
${THERMOMODELS}/phaseChangeModel/waxSolventEvaporation/waxSolventEvaporation.C

${THERMOMODELS}/heatTransferModel/heatTransferModel/heatTransferModel.C
${THERMOMODELS}/heatTransferModel/heatTransferModel/heatTransferModelNew.C
${THERMOMODELS}/heatTransferModel/constantHeatTransfer/constantHeatTransfer.C
${THERMOMODELS}/heatTransferModel/mappedConvectiveHeatTransfer/mappedConvectiveHeatTransfer.C

${THERMOMODELS}/filmRadiationModel/filmRadiationModel/filmRadiationModel.C
${THERMOMODELS}/filmRadiationModel/filmRadiationModel/filmRadiationModelNew.C
${THERMOMODELS}/filmRadiationModel/noRadiation/noRadiation.C
${THERMOMODELS}/filmRadiationModel/constantRadiation/constantRadiation.C
${THERMOMODELS}/filmRadiationModel/primaryRadiation/primaryRadiation.C
${THERMOMODELS}/filmRadiationModel/standardRadiation/standardRadiation.C

${THERMOMODELS}/filmViscosityModel/filmViscosityModel/filmViscosityModel.C
${THERMOMODELS}/filmViscosityModel/filmViscosityModel/filmViscosityModelNew.C
${THERMOMODELS}/filmViscosityModel/constantViscosity/constantViscosity.C
${THERMOMODELS}/filmViscosityModel/liquidViscosity/liquidViscosity.C
${THERMOMODELS}/filmViscosityModel/thixotropicViscosity/thixotropicViscosity.C
${THERMOMODELS}/filmViscosityModel/ArrheniusViscosity/ArrheniusViscosity.C
${THERMOMODELS}/filmViscosityModel/function1Viscosity/function1Viscosity.C
${THERMOMODELS}/filmViscosityModel/waxSolventViscosity/waxSolventViscosity.C


#/* Boundary conditions */

${PATCHFIELDS}/filmHeightInletVelocity/filmHeightInletVelocityFvPatchVectorField.C
${PATCHFIELDS}/inclinedFilmNusseltHeight/inclinedFilmNusseltHeightFvPatchScalarField.C
${PATCHFIELDS}/inclinedFilmNusseltInletVelocity/inclinedFilmNusseltInletVelocityFvPatchVectorField.C
)

# dynamic link
target_link_libraries(${PROJECT_NAME} $ENV{FOAM_LIBBIN}/libfiniteVolume.so libmeshTools.so libtriSurface.so libdynamicMesh.so libdynamicFvMesh.so
libextrudeModel.so libdecompositionMethods.so
)
target_link_libraries(${PROJECT_NAME} newdynamicMesh)
link_directories($ENV{FOAM_LIBBIN}/dummy 
)
target_link_libraries(${PROJECT_NAME}
    libcompressibleTransportModels.so 
    dfFluidThermophysicalModels
    dfThermophysicalProperties
    dfSLGThermo
    libdistributionModels.so 
    libregionModels.so
    libfiniteVolume.so 
    libmeshTools.so
)



include_directories(
    ${OpenFOAM_SRC}/transportModels/compressible/lnInclude 
    ${OpenFOAM_SRC}/thermophysicalModels/basic/lnInclude 
    ${dfthermophysicalprop_inc}
    ${OpenFOAM_SRC}/thermophysicalModels/thermophysicalProperties/lnInclude 
    ${dfthermophysicalslg}
    ${DF_SRC}/thermophysicalModels/SLGThermo/SLGThermo/
    ${CANTERA_ROOT}/include 
    ${dfcantera_inc}
    ${OpenFOAM_SRC}/lagrangian/distributionModels/lnInclude 
    ${OpenFOAM_SRC}/regionModels/regionModel/lnInclude 
    ${OpenFOAM_SRC}/regionModels/surfaceFilmModels/lnInclude 
    ${OpenFOAM_SRC}/finiteVolume/lnInclude 
    ${OpenFOAM_SRC}/meshTools/lnInclude
    
    )
    


# install
install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    )