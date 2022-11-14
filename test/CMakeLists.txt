cmake_minimum_required(VERSION 3.14)
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


file(COPY ./df0DFoam/postProcessing/probes/0/T DESTINATION 0DH2)

file(COPY ./dfHighSpeedFoam/oneD_detonationH2/postProcessing/minMax/0/fieldMinMax.dat DESTINATION 1Ddetonation)
file(COPY ./dfLowMachFoam/twoD_reactingTGV/H2/cvodeSolver/postProcessing/sample/0.0005/data_T.xy DESTINATION 2DTGV/5)
file(COPY ./dfLowMachFoam/twoD_reactingTGV/H2/cvodeSolver/postProcessing/sample/0.0004/data_T.xy DESTINATION 2DTGV/4)
file(COPY ./dfLowMachFoam/twoD_reactingTGV/H2/cvodeSolver/postProcessing/sample/0.0003/data_T.xy DESTINATION 2DTGV/3)
file(COPY ./dfLowMachFoam/twoD_reactingTGV/H2/cvodeSolver/postProcessing/sample/0.0002/data_T.xy DESTINATION 2DTGV/2)
file(COPY ./dfLowMachFoam/twoD_reactingTGV/H2/cvodeSolver/postProcessing/sample/0.0001/data_T.xy DESTINATION 2DTGV/1)


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