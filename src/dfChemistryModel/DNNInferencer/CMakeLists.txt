cmake_minimum_required(VERSION 3.0)

set(CMAKE_PREFIX_PATH $ENV{LIBTORCH_ROOT})
project(DNNInferencer)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_library(DNNInferencer SHARED DNNInferencer.cpp)
target_compile_features(DNNInferencer PUBLIC cxx_range_for)
target_link_libraries(DNNInferencer ${TORCH_LIBRARIES})
set_property(TARGET DNNInferencer PROPERTY CXX_STANDARD 14)

set(CMAKE_INSTALL_PREFIX $ENV{DF_ROOT})
install (TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    )
