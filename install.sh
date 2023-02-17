#!/bin/sh

print_finish() {
    if [ ! -z "$LIBTORCH_ROOT" ]; then
        echo " = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="
        echo "| deepflame (linked with libcantera and libtorch) compiled successfully! Enjoy!! |"
        echo " = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="
        return
    fi
    if [ ! -z "$PYTHON_LIB_DIR" ]; then
        echo " = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="
        echo "| deepflame (linked with libcantera and pytorch) compiled successfully! Enjoy!! | "
        echo " = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="
	return
    fi
    echo " = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="
    echo "| deepflame (linked with libcantera) compiled successfully! Enjoy!! |"
    echo " = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="
}





if [ ! -z $AMGX_DIR ]; then
    export AMGX_INC=$AMGX_DIR/include
    export AMGX_LIB=$AMGX_DIR/build
    export LD_LIBRARY_PATH=$FOAM2CSR_LIB:$LD_LIBRARY_PATH
    cd $DF_ROOT/foam2csr/src
    cmake -B build
    cd $DF_ROOT/foam2csr/src/build
    sudo make
    export FOAM2CSR_INC=$DF_ROOT/foam2csr/src
    export FOAM2CSR_LIB=$DF_ROOT/foam2csr/src/build
    cd $DF_ROOT/amgx4foam
    ./Allwmake
fi

./Allwmake -j && print_finish
