cmake_minimum_required(VERSION 3.10)

project(DNNInferencer_blas)

find_package(OpenMP REQUIRED)

option(YAML_INCLUDE "path to yaml include dir" /home/deepflame/yaml-cpp/include)
option(YAML_LIBRARY "path to yaml lib" /home/deepflame/yaml-cpp/build/libyaml-cpp.so)
option(BLAS_LIBRARY "path to blas lib" /home/deepflame/HPCKit/latest/kml/bisheng/lib/sme/kblas/multi/libkblas.so)
option(USE_HALF_PRECISION "open half precision" ON)

add_compile_options(-march=armv8.6-a+sve2)
add_definitions(-DENABLE_HBM)
if(USE_HALF_PRECISION)
add_definitions(-D_FP16_)
endif()

include_directories(
    ${YAML_INCLUDE}
)

add_library(${PROJECT_NAME} SHARED DNNInferencer_blas.cpp Layer.cpp kernel_double.cpp kernel_float.cpp kernel_half.cpp DNNThermo_blas.cpp)

find_library(MEMKIND_LIB memkind "path to memkind lib" /usr/lib64/libmemkind.so)
target_link_libraries(${PROJECT_NAME}
    ${YAML_LIBRARY} ${BLAS_LIBRARY} memkind
)
