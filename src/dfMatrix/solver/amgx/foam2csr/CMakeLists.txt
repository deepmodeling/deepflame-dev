cmake_minimum_required(VERSION 3.5)
project(foam_csr)
FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(MPI REQUIRED)

set(CMAKE_INSTALL_PREFIX $ENV{FOAM_USER_LIBBIN}/..)
set(AMGX_DIR $ENV{AMGX_DIR})
set ( CUDA_ENABLE true )
if ( CUDA_ENABLE )
    enable_language( CUDA )
endif()

include_directories(
./
${AMGX_DIR}/include
${MPI_INCLUDE_PATH}
${CUDA_INCLUDE_DIRS}
)

SET(CMAKE_C_COMPILER g++)
add_compile_options(-std=c++14)
# add_compile_options(-fPIC)
# add_compile_options(-arch=sm_$(NVARCH))
set(SRC_LIST AmgXCSRMatrix.cu AmgXMPIComms.cu AmgXSolver.cu)

add_library(foam_csr SHARED ${SRC_LIST})

target_link_libraries(foam_csr ${CUDA_LIBRARIES})
target_link_libraries(foam_csr ${MPI_LIBRARIES})
target_link_libraries(foam_csr ${AMGX_DIR}/build/libamgxsh.so)

install(TARGETS foam_csr DESTINATION 
PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE 
GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
