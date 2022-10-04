Installation
=================

.. Note:: If Ubuntu is used as the subsystem, please use `Ubuntu:20.04 <https://releases.ubuntu.com/focal/>`_ instead of the latest version. OpenFOAM-7 accompanied by ParaView 5.6.0 is not available for `Ubuntu-latest <https://releases.ubuntu.com/jammy/>`_.  

The installation of DeepFlame is simple and requires `OpenFOAM-7 <https://openfoam.org/version/7/>`_, `LibCantera <https://anaconda.org/conda-forge/libcantera-devel>`_, and `Libtorch <https://pytorch.org/>`_.

Install Dependencies
--------------------------

**Install OpenFOAM-7**

If `OpenFOAM-7 <https://openfoam.org/version/7/>`_ is not installed yet, please follow the instruction given on the official website. After installation, source your OpenFOAM via the default path below (or your own path for OpenFOAM bashrc).


.. code-block:: bash

    source $HOME/OpenFOAM/OpenFOAM-7/etc/bashrc

**Install LibCantera via conda**

First install Miniconda or Anaconda through commands:

.. code-block:: bash

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    sh Miniconda3-latest-Linux-x86_64.sh -b
    source ~/miniconda3/etc/profile.d/conda.sh 


Use the commands below to install and activate LibCantera.

.. code-block:: bash

    conda create -n libcantera
    conda activate libcantera
    conda install -c cantera libcantera-devel

.. Note:: Check your Miniconda3/envs/libcantera directory and make sure the install was successful (lib/ include/ etc. exist).


**Install Precompiled `Libtorch <https://pytorch.org/>`_**

.. code-block:: bash

    wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcpu.zip
    unzip libtorch-cxx11-abi-shared-with-deps-1.11.0+cpu.zip -d thirdParty


Install DeepFlame
-----------------------
**Clone the DeepFlame Repository**

.. code-block:: bash

    git clone https://github.com/deepmodeling/deepflame-dev.git
    cd deepflame-dev

**Install DeepFlame**

.. code-block:: bash

    . install.sh

.. Note:: Some compiling issues may happen due to system compatability. Instead of using conda installed Cantera C++ lib and the downloaded Torch C++ lib, try to compile your own Cantera and Torch C++ libraries.
