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
if [ $USE_LIBTORCH = true ]; then
    cd "$DF_SRC/dfChemistryModel/DNNInferencer"
    mkdir build
    cd build
    cmake ..
    if [ $? -ne 0 ]; then
        echo "Error: CMake configuration failed. Exiting installation."
        cd $DF_ROOT
        return 1
    fi
    make -j
    if [ $? -ne 0 ]; then
        echo "Error: Compilation failed. Exiting installation."
        cd $DF_ROOT
        return 1
    fi
    export LD_LIBRARY_PATH=$DF_SRC/dfChemistryModel/DNNInferencer/build:$LD_LIBRARY_PATH
fi
if [ $USE_GPUSOLVER = true ]; then
    cd "$DF_ROOT/src_gpu"
    mkdir build
    cd build
    cmake ..
    if [ $? -ne 0 ]; then
        echo "Error: CMake configuration failed. Exiting installation."
        cd $DF_ROOT
        return 1
    fi
    make -j
    if [ $? -ne 0 ]; then
        echo "Error: Compilation failed. Exiting installation."
        cd $DF_ROOT
        return 1
    fi
    export LD_LIBRARY_PATH=$DF_ROOT/src_gpu/build:$LD_LIBRARY_PATH
fi
cd $DF_ROOT
./Allwmake -j && print_finish
