#!/bin/sh

ARCH=$(uname -m)

print_finish() {
    if [ ! -z "$LIBTORCH_ROOT" ]; then
        echo " = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="
        echo "| deepflame (linked with libcantera and libtorch) in $ARCH compiled successfully! Enjoy!! |"
    elif [ ! -z "$PYTHON_LIB_DIR" ]; then
        echo " = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="
        echo "| deepflame (linked with libcantera and pytorch) in $ARCH compiled successfully! Enjoy!!  |"
    else
        echo " = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="
        echo "|     deepflame (linked with libcantera) in $ARCH compiled successfully! Enjoy!!          |"
    fi
    if [ ! -z "$AMGX_DIR" ]; then
        echo "|        select the GPU solver coupled with AMGx library to solve PDE            |"
    fi
    echo " = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="
}
if [ $USE_LIBTORCH = true ]; then
    cd "$DF_SRC/dfChemistryModel/DNNInferencer"
    mkdir build
    cd build
    cmake ..
    make 
    export LD_LIBRARY_PATH=$DF_SRC/dfChemistryModel/DNNInferencer/build:$LD_LIBRARY_PATH
fi
if [ $USE_GPUSOLVER = true ]; then
    cd "$DF_ROOT/src_gpu"
    mkdir build
    cd build
    cmake ..
    make -j
    export LD_LIBRARY_PATH=$DF_ROOT/src_gpu/build:$LD_LIBRARY_PATH
fi
cd $DF_ROOT
./Allwmake -j && print_finish
