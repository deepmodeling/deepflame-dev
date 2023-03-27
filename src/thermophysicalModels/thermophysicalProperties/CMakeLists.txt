project(dfThermophysicalProperties  LANGUAGES CXX)
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
    ${SRC_ORIG}/thermophysicalModels/thermophysicalProperties/)

set(NSRDSfunctions
    ${workDir}/thermophysicalFunctions/NSRDSfunctions)

set(SOURCES
    ${workDir}/thermophysicalFunctions/thermophysicalFunction/thermophysicalFunction.C
    
    ${NSRDSfunctions}/NSRDSfunc0/NSRDSfunc0.C
    ${NSRDSfunctions}/NSRDSfunc1/NSRDSfunc1.C
    ${NSRDSfunctions}/NSRDSfunc2/NSRDSfunc2.C
    ${NSRDSfunctions}/NSRDSfunc3/NSRDSfunc3.C
    ${NSRDSfunctions}/NSRDSfunc4/NSRDSfunc4.C
    ${NSRDSfunctions}/NSRDSfunc5/NSRDSfunc5.C
    ${NSRDSfunctions}/NSRDSfunc6/NSRDSfunc6.C
    ${NSRDSfunctions}/NSRDSfunc7/NSRDSfunc7.C
    ${NSRDSfunctions}/NSRDSfunc14/NSRDSfunc14.C
    
    ${workDir}/thermophysicalFunctions/APIfunctions/APIdiffCoefFunc/APIdiffCoefFunc.C
    
    ${workDir}/thermophysicalProperties/thermophysicalProperties.C

    ${workDir}/liquidProperties/liquidProperties/liquidProperties.C
    ${PROJECT_SOURCE_DIR}/liquidProperties/liquidMixtureProperties/liquidMixtureProperties.C
    
    ${workDir}/liquidProperties/H2O/H2O.C
    ${workDir}/liquidProperties/C7H16/C7H16.C
    ${workDir}/liquidProperties/C12H26/C12H26.C
    ${workDir}/liquidProperties/C10H22/C10H22.C
    ${workDir}/liquidProperties/C8H18/C8H18.C
    ${workDir}/liquidProperties/IC8H18/IC8H18.C
    ${workDir}/liquidProperties/C4H10O/C4H10O.C
    ${workDir}/liquidProperties/C2H6O/C2H6O.C
    ${workDir}/liquidProperties/IDEA/IDEA.C
    ${workDir}/liquidProperties/aC10H7CH3/aC10H7CH3.C
    ${workDir}/liquidProperties/bC10H7CH3/bC10H7CH3.C
    ${workDir}/liquidProperties/C8H10/C8H10.C
    ${workDir}/liquidProperties/C16H34/C16H34.C
    ${workDir}/liquidProperties/C9H20/C9H20.C
    ${workDir}/liquidProperties/C6H6/C6H6.C
    ${workDir}/liquidProperties/C7H8/C7H8.C
    ${workDir}/liquidProperties/C6H14/C6H14.C
    ${workDir}/liquidProperties/C13H28/C13H28.C
    ${workDir}/liquidProperties/C14H30/C14H30.C
    ${workDir}/liquidProperties/C3H8/C3H8.C
    ${workDir}/liquidProperties/C3H6O/C3H6O.C
    ${workDir}/liquidProperties/C2H6/C2H6.C
    ${workDir}/liquidProperties/CH3OH/CH3OH.C
    ${workDir}/liquidProperties/C2H5OH/C2H5OH.C
    ${workDir}/liquidProperties/Ar/Ar.C
    ${workDir}/liquidProperties/N2/N2.C
    ${workDir}/liquidProperties/MB/MB.C
    ${workDir}/liquidProperties/CH4N2O/CH4N2O.C
    ${workDir}/liquidProperties/nC3H8O/nC3H8O.C
    ${workDir}/liquidProperties/iC3H8O/iC3H8O.C

    ${PROJECT_SOURCE_DIR}/solidProperties/solidProperties/solidProperties.C
    ${PROJECT_SOURCE_DIR}/solidProperties/solidProperties/solidPropertiesNew.C
    ${workDir}/solidProperties/solidMixtureProperties/solidMixtureProperties.C

    ${workDir}solidProperties/ash/ash.C
    ${workDir}solidProperties/C/C.C
    ${workDir}solidProperties/CaCO3/CaCO3.C)

add_library(${PROJECT_NAME} SHARED ${SOURCES})

include_directories(
    ${PROJECT_SOURCE_DIR}/solidProperties/solidMixtureProperties
    ${PROJECT_SOURCE_DIR}/solidProperties/solidProperties
    ${workDir}
    ${OpenFOAM_SRC}/thermophysicalModels/thermophysicalProperties/lnInclude
)
message("PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR})

# install
install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    )
