Quick Start 
==============
To get a quick start with DeepFlame, there are several examples for each solver stored in the following directory that can be run.

.. code-block:: bash

    $HOME/deepflame-dev/examples

To run these examples, first source your OpenFOAM, depending on your OpenFOAM path:

.. code-block:: bash

    source $HOME/OpenFOAM/OpenFOAM-7/etc/bashrc

Then, source your DeepFlame:

.. code-block:: bash

    source $HOME/deepflame-dev/bashrc

Next, you can go to the directory of any example case that you want to run. For example:

.. code-block:: bash

    cd $HOME/deepflame-dev/examples/zeroD_cubicReactor/H2/cvodeSolver

This is an example for the zero-dimensional hydrogen combustion  with CVODE Solver.

The case is run by simply typing: 

.. code-block:: bash

    ./Allrun

.. Note:: For the example cases with torchSolver, an additional DNN model file in the `.pt` format is required. Please contact the developers if you would like a test run.



