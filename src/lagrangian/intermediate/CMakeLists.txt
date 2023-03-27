project(dfLagrangianIntermediate LANGUAGES CXX)
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
    ${OpenFOAM_SRC}/lagrangian/distributionModels/lnInclude 
    ${dflagrangianinter_inc} 
    ${OpenFOAM_SRC}/lagrangian/intermediate/lnInclude 
    ${OpenFOAM_SRC}/transportModels/compressible/lnInclude 
    ${OpenFOAM_SRC}/thermophysicalModels/basic/lnInclude 
    ${dfthermophysicalprop_inc} 
    ${OpenFOAM_SRC}/thermophysicalModels/thermophysicalProperties/lnInclude 
    ${dfthermophysicalslg_inc} 
    $ENV{CANTERA_ROOT}/include
    ${dfcantera_inc} 
    ${OpenFOAM_SRC}/transportModels 
    ${OpenFOAM_SRC}/regionModels/regionModel/lnInclude 
    ${OpenFOAM_SRC}/regionModels/surfaceFilmModels/lnInclude 
    ${OpenFOAM_SRC}/dynamicFvMesh/lnInclude 
    ${OpenFOAM_SRC}/sampling/lnInclude 
    ${OpenFOAM_SRC}/finiteVolume/lnInclude 
    ${OpenFOAM_SRC}/meshTools/lnInclude

)

set(workDir ${SRC_ORIG}/lagrangian/intermediate)
set(PARCELS ${CMAKE_CURRENT_SOURCE_DIR}/parcels)
set(BASEPARCELS ${PARCELS}/baseClasses)
set(DERIVEDPARCELS ${PARCELS}/derived)

set(CLOUDS ${CMAKE_CURRENT_SOURCE_DIR}/clouds)
set(BASECLOUDS ${CLOUDS}/baseClasses)
set(DERIVEDCLOUDS ${CLOUDS}/derived)
set(KINEMATICPARCEL ${DERIVEDPARCELS}/basicKinematicParcel)
set(KINEMATICCOLLIDINGPARCEL  ${DERIVEDPARCELS}/basicKinematicCollidingParcel)
set(THERMOPARCEL  ${DERIVEDPARCELS}/basicThermoParcel)
set(REACTINGPARCEL  ${DERIVEDPARCELS}/basicReactingParcel)
set(REACTINGMPPARCEL  ${DERIVEDPARCELS}/basicReactingMultiphaseParcel)
set(KINEMATICMPPICPARCEL  ${DERIVEDPARCELS}/basicKinematicMPPICParcel)
set(KINEMATICINJECTION  ${workDir}/submodels/Kinematic/InjectionModel)
set(THERMOINJECTION  ${workDir}/submodels/Thermodynamic/InjectionModel)
set(REACTINGMPINJECTION  ${workDir}/submodels/ReactingMultiphase/InjectionModel)
set(REACTINGINJECTION  ${workDir}/submodels/Reacting/InjectionModel)
set(MPPICCORRECTIONLIMITING  ${workDir}/submodels/MPPIC/CorrectionLimitingMethods)
set(MPPICPARTICLESTRESS  ${workDir}/submodels/MPPIC/ParticleStressModels)
set(MPPICCORRECTIONLIMITING  ${workDir}/submodels/MPPIC/CorrectionLimitingMethods)
set(MPPICTIMESCALE  ${workDir}/submodels/MPPIC/TimeScaleModels)
add_library(dfLagrangianIntermediate SHARED


${BASECLOUDS}/kinematicCloud/kinematicCloud.C
${BASECLOUDS}/thermoCloud/thermoCloud.C
${BASECLOUDS}/reactingCloud/reactingCloud.C
${BASECLOUDS}/reactingMultiphaseCloud/reactingMultiphaseCloud.C

${KINEMATICPARCEL}/defineBasicKinematicParcel.C
${KINEMATICPARCEL}/makeBasicKinematicParcelSubmodels.C

${KINEMATICCOLLIDINGPARCEL}/defineBasicKinematicCollidingParcel.C
${KINEMATICCOLLIDINGPARCEL}/makeBasicKinematicCollidingParcelSubmodels.C



${THERMOPARCEL}/defineBasicThermoParcel.C
${THERMOPARCEL}/makeBasicThermoParcelSubmodels.C



${REACTINGPARCEL}/defineBasicReactingParcel.C
${REACTINGPARCEL}/makeBasicReactingParcelSubmodels.C



${REACTINGMPPARCEL}/defineBasicReactingMultiphaseParcel.C
${REACTINGMPPARCEL}/makeBasicReactingMultiphaseParcelSubmodels.C



${KINEMATICMPPICPARCEL}/defineBasicKinematicMPPICParcel.C
${KINEMATICMPPICPARCEL}/makeBasicKinematicMPPICParcelSubmodels.C



${workDir}/submodels/Kinematic/PatchInteractionModel/LocalInteraction/patchInteractionData.C
${workDir}/submodels/Kinematic/PatchInteractionModel/LocalInteraction/patchInteractionDataList.C


${KINEMATICINJECTION}/KinematicLookupTableInjection/kinematicParcelInjectionData.C
${KINEMATICINJECTION}/KinematicLookupTableInjection/kinematicParcelInjectionDataIO.C
${KINEMATICINJECTION}/KinematicLookupTableInjection/kinematicParcelInjectionDataIOList.C
${KINEMATICINJECTION}/PatchInjection/patchInjectionBase.C


${THERMOINJECTION}/ThermoLookupTableInjection/thermoParcelInjectionData.C
${THERMOINJECTION}/ThermoLookupTableInjection/thermoParcelInjectionDataIO.C
${THERMOINJECTION}/ThermoLookupTableInjection/thermoParcelInjectionDataIOList.C


${REACTINGINJECTION}/ReactingLookupTableInjection/reactingParcelInjectionData.C
${REACTINGINJECTION}/ReactingLookupTableInjection/reactingParcelInjectionDataIO.C
${REACTINGINJECTION}/ReactingLookupTableInjection/reactingParcelInjectionDataIOList.C


${REACTINGMPINJECTION}/ReactingMultiphaseLookupTableInjection/reactingMultiphaseParcelInjectionData.C
${REACTINGMPINJECTION}/ReactingMultiphaseLookupTableInjection/reactingMultiphaseParcelInjectionDataIO.C
${REACTINGMPINJECTION}/ReactingMultiphaseLookupTableInjection/reactingMultiphaseParcelInjectionDataIOList.C


${MPPICPARTICLESTRESS}/ParticleStressModel/ParticleStressModel.C
${MPPICPARTICLESTRESS}/HarrisCrighton/HarrisCrighton.C
${MPPICPARTICLESTRESS}/Lun/Lun.C
${MPPICPARTICLESTRESS}/exponential/exponential.C


${MPPICCORRECTIONLIMITING}/CorrectionLimitingMethod/CorrectionLimitingMethod.C
${MPPICCORRECTIONLIMITING}/noCorrectionLimiting/noCorrectionLimiting.C
${MPPICCORRECTIONLIMITING}/absolute/absolute.C
${MPPICCORRECTIONLIMITING}/relative/relative.C


${MPPICTIMESCALE}/TimeScaleModel/TimeScaleModel.C
${MPPICTIMESCALE}/equilibrium/equilibrium.C
${MPPICTIMESCALE}/nonEquilibrium/nonEquilibrium.C
${MPPICTIMESCALE}/isotropic/isotropic.C


${workDir}/integrationScheme/integrationScheme/integrationScheme.C
${workDir}/integrationScheme/integrationScheme/integrationSchemeNew.C
${workDir}/integrationScheme/Euler/Euler.C
${workDir}/integrationScheme/analytical/analytical.C


${workDir}/phaseProperties/phaseProperties/phaseProperties.C
${workDir}/phaseProperties/phaseProperties/phasePropertiesIO.C
${workDir}/phaseProperties/phasePropertiesList/phasePropertiesList.C


${CMAKE_CURRENT_SOURCE_DIR}/clouds/Templates/KinematicCloud/cloudSolution/cloudSolution.C



${workDir}/submodels/MPPIC/AveragingMethods/makeAveragingMethods.C
)


# dynamic link
target_link_libraries(${PROJECT_NAME} $ENV{FOAM_LIBBIN}/libcompressibleTransportModels.so libregionModels.so
libfiniteVolume.so libmeshTools.so liblagrangian.so libdistributionModels.so libincompressibleTransportModels.so libdynamicFvMesh.so libsampling.so
libmeshTools.so
)

target_link_libraries(${PROJECT_NAME} dfCompressibleTurbulenceModels
dfCanteraMixture
dfChemistryModel
dfFluidThermophysicalModels
dfThermophysicalProperties
dfSLGThermo
dfSurfaceFilmModels
)

target_link_libraries(${PROJECT_NAME} ${CANTERA_ROOT}/lib/libcantera_shared.so.2)



install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    )
