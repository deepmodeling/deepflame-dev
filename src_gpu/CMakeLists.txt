#
# dfMatrix CMake configuration
#
cmake_minimum_required(VERSION 3.5)

project(dfMatrix LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_PREFIX_PATH /root/libtorch/libtorch)

find_package(CUDA REQUIRED)
find_package(MPI REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)
find_library(LIBAMGXSH amgxsh PATHS $ENV{AMGX_DIR}/build)

add_compile_options(-arch=sm_70 -fmad=false)

include_directories(
    ${MPI_INCLUDE_PATH}
    ${CUDA_INCLUDE_DIRS}
    $ENV{AMGX_DIR}/include
    ${CMAKE_SOURCE_DIR}
    dfLinearSolver/CSRMatrix
    dfLinearSolver/ELLMatrix
)

add_library(${PROJECT_NAME} 
    SHARED
        AmgXSolver.cu
        dfMatrixDataBase.cu
        dfMatrixOpBase.cu
        dfNcclBase.cu
        dfUEqn.cu
        dfYEqn.cu
        dfEEqn.cu
        dfRhoEqn.cu
        dfpEqn.cu
        dfThermo.cu
        dfChemistrySolver.cu

        dfLinearSolver/CSRMatrix/dfPBiCGStabCSRSolver.cu)

target_link_libraries(${PROJECT_NAME}
    ${MPI_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${LIBAMGXSH}
    ${TORCH_LIBRARIES}
)
target_compile_options(dfMatrix PUBLIC -g)
option(DFMATRIX_ENABLE_DETAILED_DEBUG "Enable detailed debug build" OFF)
if (DFMATRIX_ENABLE_DETAILED_DEBUG)
    target_compile_definitions(${PROJECT_NAME} PRIVATE DEBUG_)
endif()

# target_compile_definitions(${PROJECT_NAME} PRIVATE STREAM_ALLOCATOR)

