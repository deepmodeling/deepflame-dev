#!/bin/sh

if [ -d "thirdParty/libtorch" ]; then
    echo "libtorch exist."
else

    if [ -e libtorch-cxx11-abi-shared-with-deps-1.11.0+cpu.zip ]
    then
        echo "libtorch.zip exist."
    else
        wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcpu.zip
    fi

    unzip libtorch-cxx11-abi-shared-with-deps-1.11.0+cpu.zip -d thirdParty
fi


cp bashrc.raw bashrc
sed -i s#pwd#$PWD#g ./bashrc
sed -i s#CONDA_PREFIX#$CONDA_PREFIX#g ./bashrc
TORCH_DIR=$PWD/thirdParty/libtorch
sed -i s#TORCH_DIR#$TORCH_DIR#g ./bashrc



if [ -d "src_orig" ]; then
    echo "src_orig exist."
else
    mkdir -p src_orig/TurbulenceModels
    mkdir -p src_orig/thermophysicalModels
    mkdir -p src_orig/lagrangian
    mkdir -p src_orig/regionModels
    mkdir -p src_orig/functionObjects
    cp -r $FOAM_SRC/TurbulenceModels/compressible src_orig/TurbulenceModels
    cp -r $FOAM_SRC/thermophysicalModels/basic src_orig/thermophysicalModels
    cp -r $FOAM_SRC/thermophysicalModels/thermophysicalProperties src_orig/thermophysicalModels
    cp -r $FOAM_SRC/lagrangian/intermediate src_orig/lagrangian
    cp -r $FOAM_SRC/lagrangian/turbulence src_orig/lagrangian
    cp -r $FOAM_SRC/regionModels/surfaceFilmModels src_orig/regionModels
    cp -r $FOAM_SRC/functionObjects/field src_orig/functionObjects
fi


source ./bashrc
./Allwmake -j