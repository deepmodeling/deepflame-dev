project(dfFluidThermophysicalModels  LANGUAGES CXX)
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
PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    )
