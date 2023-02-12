project(dfDynamicFvMesh LANGUAGES CXX)

# add library
add_library(dfDynamicFvMesh SHARED

${DF_SRC}/dynamicFvMesh/dfDynamicRefineFvMesh/multiCritRefinement.C
${DF_SRC}/dynamicFvMesh/dfDynamicRefineFvMesh/dfDynamicRefineFvMesh.C

${DF_SRC}/dynamicFvMesh/dfRefinementHistoryConstraint/dfRefinementHistoryConstraint.C
)

# dynamic link
target_link_libraries(${PROJECT_NAME} $ENV{FOAM_LIBBIN}/libfiniteVolume.so libmeshTools.so libtriSurface.so libdynamicMesh.so libdynamicFvMesh.so
libextrudeModel.so libdecompositionMethods.so
)
target_link_libraries(${PROJECT_NAME} newdynamicMesh)
link_directories($ENV{FOAM_LIBBIN}/dummy 
)
target_link_libraries(${PROJECT_NAME}
                      $ENV{FOAM_LIBBIN}/dummy/libscotchDecomp.so
                      $ENV{FOAM_LIBBIN}/dummy/libptscotchDecomp.so
                      $ENV{FOAM_LIBBIN}/dummy/libmetisDecomp.so
)



include_directories(
    ${dffvdynamic_inc}
    ${dfnewdynamic_inc}
    ${OpenFOAM_SRC}/meshTools/lnInclude
    ${OpenFOAM_SRC}/triSurface/lnInclude
    ${OpenFOAM_SRC}/dynamicFvMesh/lnInclude
    ${OpenFOAM_SRC}/dynamicMesh/lnInclude
    ${OpenFOAM_SRC}/parallel/decompose/decompositionMethods/lnInclude
    
    )
    


# install
install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    )