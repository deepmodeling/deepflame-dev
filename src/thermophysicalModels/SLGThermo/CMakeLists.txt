project(dfSLGThermo  LANGUAGES CXX)

# add library

add_library(${PROJECT_NAME} SHARED 
${CMAKE_CURRENT_SOURCE_DIR}/SLGThermo/SLGThermo.C)

include_directories(

    ${OpenFOAM_SRC}/transportModels/compressible/lnInclude 
    ${OpenFOAM_SRC}/thermophysicalModels/basic/lnInclude 
    ${dfthermophysicalprop_inc}
    ${OpenFOAM_SRC}/thermophysicalModels/thermophysicalProperties/lnInclude 
    ${CANTERA_ROOT}/include
    ${dfcantera_inc}
    ${OpenFOAM_SRC}/finiteVolume/lnInclude
)



target_link_libraries(${PROJECT_NAME} $ENV{FOAM_LIBBIN}/libfiniteVolume.so libmeshTools.so libcompressibleTransportModels.so)

target_link_libraries(${PROJECT_NAME} ${CANTERA_ROOT}/lib/libcantera_shared.so.2)

target_link_libraries(${PROJECT_NAME} 

dfThermophysicalProperties
dfCanteraMixture

)

# install
install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    )
