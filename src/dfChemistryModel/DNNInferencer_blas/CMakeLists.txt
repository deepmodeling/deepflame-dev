cmake_minimum_required(VERSION 3.10)

project(DNNInferencer_blas)

find_package(OpenMP REQUIRED)

option(YAML_INCLUDE "path to yaml include dir" "")
option(YAML_LIBRARY "path to yaml lib" "")
option(BLAS_LIBRARY "path to blas lib" "")

option(USE_HALF_PRECISION "open half precision" OFF)

if(USE_HALF_PRECISION)
add_definitions(-D_FP16_)
endif()

include_directories(
    ${YAML_INCLUDE}
)

add_library(${PROJECT_NAME} SHARED DNNInferencer_blas.cpp Layer.cpp kernel_double.cpp kernel_float.cpp kernel_half.cpp DNNThermo_blas.cpp)

target_link_libraries(${PROJECT_NAME}   
    ${YAML_LIBRARY} ${BLAS_LIBRARY}
)
