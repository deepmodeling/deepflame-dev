#!/bin/sh

print_finish() {
    if [ ! -z "$LIBTORCH_ROOT" ]; then
        echo " = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="
        echo "| deepflame (linked with libcantera and libtorch) compiled successfully! Enjoy!! |"
    elif [ ! -z "$PYTHON_LIB_DIR" ]; then
        echo " = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="
        echo "| deepflame (linked with libcantera and pytorch) compiled successfully! Enjoy!!  |"
    else
        echo " = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="
        echo "|     deepflame (linked with libcantera) compiled successfully! Enjoy!!          |"
    fi
    if [ ! -z "$AMGX_DIR" ]; then
        echo "|        select the GPU solver coupled with AMGx library to solve PDE            |"
    fi
    echo " = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="
}
# if [ $USE_LIBTORCH = true ]; then
#     cd "$DF_SRC/dfChemistryModel/DNNInferencer"
#     mkdir build
#     cd build
#     cmake ..
#     make 
#     export LD_LIBRARY_PATH=$DF_SRC/dfChemistryModel/DNNInferencer/build:$LD_LIBRARY_PATH
# fi
# if [ $USE_GPUSOLVER = true ]; then
#     cd "$DF_ROOT/src_gpu"
#     mkdir build
#     cd build
#     cmake ..
#     make -j
#     export LD_LIBRARY_PATH=$DF_ROOT/src_gpu/build:$LD_LIBRARY_PATH
# fi

echo  USE_BLASDNN : $USE_BLASDNN

if [ $USE_BLASDNN == true ]; then
    cd "$DF_ROOT"
    mkdir -p lib
    cd "$DF_SRC/dfChemistryModel/DNNInferencer_blas"
    mkdir -p build
    cd build
    cmake .. \
        -DCMAKE_CXX_COMPILER=mpiCC \
        -DCMAKE_CXX_FLAGS="-Ofast -g -fopenmp -std=c++11" \
        -DUSE_HALF_PRECISION=ON \
        -DYAML_INCLUDE=/home/deepflame/yaml-cpp/include \
        -DYAML_LIBRARY=/home/deepflame/yaml-cpp/build/libyaml-cpp.so \
        -DBLAS_LIBRARY=/home/deepflame/HPCKit/latest/kml/bisheng/lib/sme/kblas/multi/libkblas.so

    if [ $? -ne 0 ]; then
        echo "Error: CMake configuration failed. Exiting installation."
        cd $DF_ROOT
        return 1
    fi
    make VERBOSE=1
    if [ $? -ne 0 ]; then
        echo "Error: Compilation failed. Exiting installation."
        cd $DF_ROOT
        return 1
    fi
    cp ./libDNNInferencer_blas.so $DF_ROOT/lib/
fi

cd $DF_ROOT
./Allwmake -j && print_finish
