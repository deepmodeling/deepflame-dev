cmake_minimum_required(VERSION 3.10)

project(DNNInferencer_blas)

find_package(OpenMP REQUIRED)

set(YAML_INCLUDE "$ENV{YAML_ROOT}/yaml-cpp/include")
set(YAML_LIBRARY "$ENV{YAML_ROOT}/lib64/libyaml-cpp.so")
set(BLAS_LIBRARY "$ENV{HPCKIT_ROOT}/kml/bisheng/lib/sme/kblas/multi/libkblas.so")
set(USE_HALF_PRECISION "open half precision" ON)

add_compile_options(-march=armv8.6-a+sve2)
add_definitions(-DENABLE_HBM)
if(USE_HALF_PRECISION)
add_definitions(-D_FP16_)
endif()

add_library(${PROJECT_NAME} SHARED DNNInferencer_blas.cpp Layer.cpp kernel_double.cpp kernel_float.cpp kernel_half.cpp DNNThermo_blas.cpp)

message("$ENV{YAML_ROOT}/include/yaml-cpp $ENV{HPCKIT_ROOT}/hmpi/bisheng/release/hmpi/include")
target_include_directories(${PROJECT_NAME}
PUBLIC
    $ENV{YAML_ROOT}/include $ENV{HPCKIT_ROOT}/hmpi/bisheng/release/hmpi/include /usr/include/sys/mman.h
)


#set(MEMKIND_LIB memkind "/usr/lib64/libmemkind.so")
target_link_libraries(${PROJECT_NAME}
    $ENV{YAML_ROOT}/lib64/libyaml-cpp.so $ENV{HPCKIT_ROOT}/kml/bisheng/lib/sme/kblas/multi/libkblas.so
)
