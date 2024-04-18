cmake_minimum_required(VERSION 3.5)
project(dfMatrix)

include_directories(
    ${PROJECT_SOURCE_DIR}
)

SET(CMAKE_C_COMPILER g++)
add_compile_options(-std=c++17 -g)

add_library(${PROJECT_NAME} SHARED dfMatrix.cpp)