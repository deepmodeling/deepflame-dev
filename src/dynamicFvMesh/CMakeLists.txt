project(dfDynamicFvMesh LANGUAGES CXX)

# Compiling configure
add_definitions("${SHARED_COMPILE_OPTION}")

# add library
SET(workDir ${DF_SRC}/dynamicFvMesh)

SET(SOURCES
${workDir}/dfDynamicRefineFvMesh/multiCritRefinement.C
${workDir}/dfDynamicRefineFvMesh/dfDynamicRefineFvMesh.C
${workDir}/dfRefinementHistoryConstraint/dfRefinementHistoryConstraint.C
)

add_library(dfDynamicFvMesh SHARED ${SOURCES})

# dynamic link
target_link_libraries(${PROJECT_NAME} ${OpenFOAM_LIB_DIR}/libfiniteVolume.so libmeshTools.so
libdynamicMesh.so
libdynamicFvMesh.so
libextrudeModel.so
libdecompositionMethods.so
newdynamicMesh
)

link_directories(${OpenFOAM_LIB_DIR}/dummy
)
target_link_libraries(${PROJECT_NAME}
                      ${OpenFOAM_LIB_DIR}/dummy/libscotchDecomp.so
                      ${OpenFOAM_LIB_DIR}/dummy/libptscotchDecomp.so
                      ${OpenFOAM_LIB_DIR}/dummy/libmetisDecomp.so
)



include_directories(
    ${dffvdynamic_inc}
    ${dfnewdynamic_inc}
    ${OpenFOAM_SRC}/meshTools/lnInclude
    ${OpenFOAM_SRC}/dynamicFvMesh/lnInclude
    ${OpenFOAM_SRC}/dynamicMesh/lnInclude
    ${OpenFOAM_SRC}/parallel/decompose/decompositionMethods/lnInclude

    )



# install
install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)