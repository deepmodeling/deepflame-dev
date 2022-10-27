
Install OpenFOAM-7 (if not already installed)
====================================================

.. Note:: If Ubuntu is used as the subsystem, please use `Ubuntu:20.04 <https://releases.ubuntu.com/focal/>`_ instead of the latest version. OpenFOAM-7 accompanied by ParaView 5.6.0 is not available for `Ubuntu-latest <https://releases.ubuntu.com/jammy/>`_.  

.. code-block:: bash

    sudo sh -c "wget -O - https://dl.openfoam.org/gpg.key | apt-key add -"
    sudo add-apt-repository http://dl.openfoam.org/ubuntu
    sudo apt-get update
    sudo apt-get -y install openfoam7

OpenFOAM 7 and ParaView 5.6.0 will be installed in the ``/opt`` directory.

Source your OpenFOAM
======================

.. code-block:: bash

    source $HOME/OpenFOAM/OpenFOAM-7/etc/bashrc

This depends on your own path for OpenFOAM bashrc.

Clone the DeepFlame repository
===========================================

.. code-block:: bash

    git clone https://github.com/deepmodeling/deepflame-dev.git
    cd deepflame-dev


Install dependencies and DeepFlame based on your need
=================================================================
DeepFlame supports three compilation choices: no torch, LibTorch, and PyTorch.

    .. Note:: You are encouaged to try all three options, but remember to install the next version in a new terminal to clean previous environment variables.

No Torch version
-------------------------

If your are using DeepFlame's CVODE solver without DNN model, just install LibCantera via `conda <https://docs.conda.io/en/latest/miniconda.html#linux-installers>`_.

.. code-block:: bash

    conda create -n df-notorch
    conda activate df-notorch
    conda install -c cantera libcantera-devel

.. Note:: Check your ``Miniconda3/envs/libcantera`` directory and make sure the install was successful (lib/ include/ etc. exist).


If the conda env ``df-notorch`` is activated, install DeepFlame by running:

.. code-block:: bash

    . install.sh 

If ``df-notorch`` not activated (or you have a self-complied libcantera), specify the path to your libcantera:

.. code-block:: bash

    . install.sh --libcantera_dir /your/path/to/libcantera/



LibTorch version
-------------------------------

If you choose to use LibTorch (C++ API for Torch), first create the conda env and install `LibCantera <https://anaconda.org/conda-forge/libcantera-devel>`_:
    
.. code-block:: bash

    conda create -n df-libtorch
    conda activate df-libtorch
    conda install -c cantera libcantera-devel

Then you can either install DeepFlame with autodownloaded LibTorch

.. code-block:: bash

    . install.sh --libtorch_autodownload

or you can pass your own libtorch path to DeepFlame.

.. code-block:: bash

    . install.sh --libtorch_dir /path/to/libtorch/


PyTorch version
-------------------------------

PyTorch version aims to support computation on CUDA. If you have compatible platform, run the following command to install DeepFlame.

.. code-block:: 

    conda create -n df-pytorch python=3.8
    conda activate df-pytorch
    conda install -c cantera libcantera-devel
    conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge 
    conda install pybind11
    . install.sh --use_pytorch

.. Note:: You may come accross an error regarding shared library ``libmkl_rt.so.2`` when libcantera is installed through cantera channel. If so, go to your conda environment and check the existance of ``libmkl_rt.so.2`` and ``libmkl_rt.so.1``, and then link ``libmkl_rt.so.2`` to ``libmkl_rt.so.1``.
    

.. code-block:: bash

    cd ~/miniconda3/envs/df-pytorch/lib
    ln -s libmkl_rt.so.1 libmkl_rt.so.2


.. Note::  Some compiling issues may happen due to system compatability. Instead of using conda installed Cantera C++ lib and the downloaded Torch C++ lib, try to compile your own Cantera and Torch C++ libraries.

