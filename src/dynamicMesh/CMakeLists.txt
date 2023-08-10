project(newdynamicMesh LANGUAGES CXX)

# Compiling configure
add_definitions("${SHARED_COMPILE_OPTION}")

# add library

SET(hexRef ${DF_SRC}/dynamicMesh/polyTopoChange/polyTopoChange/hexRef)

SET(SOURCES
${hexRef}/hexRef.C
${hexRef}/hexRefData.C
${hexRef}/dfRefinementHistory.C
${hexRef}/hexRefNew.C
${hexRef}/hexRef8/hexRef8.C
${hexRef}/hexRef4/hexRef4.C
${hexRef}/hexRef4Axi/hexRef4Axi.C
${hexRef}/hexRef2/hexRef2.C
)

add_library(newdynamicMesh SHARED ${SOURCES})

# dynamic link
target_link_libraries(${PROJECT_NAME} ${OpenFOAM_LIB_DIR}/libfiniteVolume.so
libmeshTools.so
libdynamicMesh.so
libdynamicFvMesh.so
libextrudeModel.so
libdecompositionMethods.so
)


link_directories($ENV{FOAM_LIBBIN}/dummy)
target_link_libraries(${PROJECT_NAME} ${OpenFOAM_LIB_DIR}/dummy/libscotchDecomp.so
                      ${OpenFOAM_LIB_DIR}/dummy/libptscotchDecomp.so
                      ${OpenFOAM_LIB_DIR}/dummy/libmetisDecomp.so
)



include_directories(
    ${dfnewdynamic_inc}
    ${OpenFOAM_SRC}/meshTools/lnInclude
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