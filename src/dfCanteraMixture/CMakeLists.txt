
project(dfCanteraMixture LANGUAGES CXX)

include_directories(
                    ./
                    ${OpenFOAM_SRC}/meshTools/lnInclude
                    ${OpenFOAM_SRC}/thermophysicalModels/basic/lnInclude
                    ${OpenFOAM_SRC}/transportModels/compressible/lnInclude
                    ${CANTERA_ROOT}/include
                    )




add_library(dfCanteraMixture SHARED
    CanteraMixture.C
    makeThermos.C
)


# dynamic link
target_link_libraries(${PROJECT_NAME} $ENV{FOAM_LIBBIN}/libfiniteVolume.so libmeshTools.so)

target_link_libraries(${PROJECT_NAME} ${CANTERA_ROOT}/lib/libcantera_shared.so.2)




install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    )
