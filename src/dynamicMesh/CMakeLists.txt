project(newdynamicMesh LANGUAGES CXX)

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
    )