cmake_minimum_required(VERSION 3.18)
project(Test)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


file(COPY ./df0DFoam/zeroD_cubicReactor/H2/cvodeSolver/postProcessing/probes/0/T DESTINATION 0DH2)
file(COPY ./df0DFoam/zeroD_cubicReactor/CH4/cvodeSolver/postProcessing/probes/0/T DESTINATION 0DCH4)
file(COPY ./dfLowMachFoam/twoD_reactingTGV/H2/cvodeSolver/postProcessing/sample/0.0005/data_T.xy DESTINATION 2DTGV)


enable_testing()

add_executable(
  corrtest
  corrtest.cpp
)
target_link_libraries(
  corrtest
  GTest::gtest_main
)


include(GoogleTest)
gtest_discover_tests(corrtest) 