project(newdynamicMesh LANGUAGES CXX)
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
add_library(newdynamicMesh SHARED

${DF_SRC}/dynamicMesh/polyTopoChange/polyTopoChange/hexRef/hexRef.C
${DF_SRC}/dynamicMesh/polyTopoChange/polyTopoChange/hexRef/hexRefData.C
${DF_SRC}/dynamicMesh/polyTopoChange/polyTopoChange/hexRef/dfRefinementHistory.C
${DF_SRC}/dynamicMesh/polyTopoChange/polyTopoChange/hexRef/hexRefNew.C
${DF_SRC}/dynamicMesh/polyTopoChange/polyTopoChange/hexRef/hexRef8/hexRef8.C
${DF_SRC}/dynamicMesh/polyTopoChange/polyTopoChange/hexRef/hexRef4/hexRef4.C
${DF_SRC}/dynamicMesh/polyTopoChange/polyTopoChange/hexRef/hexRef4Axi/hexRef4Axi.C
${DF_SRC}/dynamicMesh/polyTopoChange/polyTopoChange/hexRef/hexRef2/hexRef2.C

)

# dynamic link
target_link_libraries(${PROJECT_NAME} $ENV{FOAM_LIBBIN}/libfiniteVolume.so libmeshTools.so libtriSurface.so libdynamicMesh.so libdynamicFvMesh.so
libextrudeModel.so libdecompositionMethods.so
)


link_directories($ENV{FOAM_LIBBIN}/dummy)
target_link_libraries(${PROJECT_NAME} $ENV{FOAM_LIBBIN}/dummy/libscotchDecomp.so
                      $ENV{FOAM_LIBBIN}/dummy/libptscotchDecomp.so
                      $ENV{FOAM_LIBBIN}/dummy/libmetisDecomp.so
)



include_directories(
    ${dfnewdynamic_inc}
    ${OpenFOAM_SRC}/meshTools/lnInclude
    ${OpenFOAM_SRC}/triSurface/lnInclude
    ${OpenFOAM_SRC}/dynamicFvMesh/lnInclude
    ${OpenFOAM_SRC}/dynamicMesh/lnInclude
    ${OpenFOAM_SRC}/dynamicMesh/lnInclude
    ${OpenFOAM_SRC}/parallel/decompose/decompositionMethods/lnInclude
    ${OpenFOAM_SRC}/mesh/extrudeModel/lnInclude
    
    )
    


# install
install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    )