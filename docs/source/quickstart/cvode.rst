CVODE Solver
===================

Follow the steps below to run an example of CVODE. Examples are stored in the directory: 
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

This is an example for the zero-dimensional hydrogen combustion  with CVODE integrator.

The case is run by simply typing: 

.. code-block:: bash

    ./Allrun

The probe used for post processing is defined in ``/system/probes``. In this case, the probe is located at the coordinates (0.0025 0.0025 0.0025) to measure temperature variation with time. 
If the case is successfully run, the result can be found in ``/postProcessing/probes/0/T``, and it can be visualized by running: 

.. code-block:: bash

    gunplot
    plot "/your/path/to/postProcessing/probes/0/T"

You will get a graph:

.. figure:: 0Dcvode.jpg
    
    Visualisation of the zero-dimensional hydrogen combustion result with CVODE solver