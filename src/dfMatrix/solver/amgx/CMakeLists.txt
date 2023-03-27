cmake_minimum_required(VERSION 3.5)
project(AMGX)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(MPI REQUIRED)

# set variables
set(CMAKE_INSTALL_PREFIX $ENV{FOAM_USER_LIBBIN}/..)
set(AMGX_DIR $ENV{AMGX_DIR})
set(OpenFOAM_SRC $ENV{FOAM_SRC})
set ( CUDA_ENABLE true )
if ( CUDA_ENABLE )
    enable_language( CUDA )
endif()

# add library foam2csr
set(SRC_LIST 
foam2csr/AmgXCSRMatrix.cu 
foam2csr/AmgXMPIComms.cu 
foam2csr/AmgXSolver.cu)
add_library(foam_csr SHARED ${SRC_LIST})

target_include_directories(foam_csr
    PUBLIC
        foam2csr
        ${MPI_INCLUDE_PATH}
        ${CUDA_INCLUDE_DIRS}
        ${AMGX_DIR}/include
)
target_link_libraries(foam_csr
    PUBLIC    
        ${CUDA_LIBRARIES}
        ${MPI_LIBRARIES}
        ${AMGX_DIR}/build/libamgxsh.so
)

# add library AMGX
set(SOURCE
    utils/linearSolverContextTable.C
    utils/amgxLinearSolverContext.C
    utils/cacheManager.C
    amgxSolver.C)

add_library(${PROJECT_NAME} SHARED ${SOURCE})

set(LINK_FLAGS  "-Xlinker --add-needed -Xlinker --no-as-needed")
set(CMAKE_SHARED_LINKER_FLAGS "${LINK_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${LINK_FLAGS}")

set(DEFINITIONS_COMPILE "-std=c++11 -DWM_ARCH_OPTION=64 -DWM_DP 
-DWM_LABEL_SIZE=32 -Wall -Wextra -Wno-unused-parameter  
-Wno-old-style-cas -Wno-overloaded-virtual -Wno-unused-variable 
-Wno-unused-local-typedef -Wno-invalid-offsetof -Wno-deprecated-register 
-Wno-undefined-var-template -O0 -g -DFULLDEBUG -DNoRepository 
-ftemplate-depth-100 -fPIC")

target_compile_options(${PROJECT_NAME}
    PRIVATE
        -std=c++11 -DWM_ARCH_OPTION=64 -DWM_DP 
        -DWM_LABEL_SIZE=32 -Wall -Wextra -Wno-unused-parameter  
        -Wno-overloaded-virtual -Wno-unused-local-typedef 
        -Wno-invalid-offsetof -O3 -g -DFULLDEBUG -DNoRepository 
        -ftemplate-depth-100 -fPIC)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        foam2csr
        utils
        ${OpenFOAM_SRC}/finiteVolume/lnInclude
        ${OpenFOAM_SRC}/OSspecific/POSIX/lnInclude
        ${OpenFOAM_SRC}/OpenFOAM/lnInclude
        ${OpenFOAM_SRC}/meshTools/lnInclude
        ${OpenFOAM_SRC}/Pstream/mpi
)

SET(CMAKE_C_COMPILER g++)
add_compile_options(-std=c++14)

target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${MPI_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${AMGX_DIR}/build/libamgxsh.so)
target_link_libraries(${PROJECT_NAME} $ENV{FOAM_LIBBIN}/libfiniteVolume.so libmeshTools.so)
target_link_libraries(${PROJECT_NAME} foam_csr)

# install(TARGETS foam_csr DESTINATION 
# PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE 
# GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
