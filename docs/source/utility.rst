Reaction Mechanism Convertion
=================================
DeepFlame uses *yaml* reaction mechanisms, which are compatible with Cantera. The following command lines can be used to convert *chemkin* mechanisms into *yaml* format. 


.. code-block:: bash

    conda create --name ct-env --channel conda-forge cantera 
    conda activate ct-env
    ck2yaml --input=chem.inp --thermo=therm.dat --transport=tran.dat

.. Note:: Users will need to create a new conda environment other than the one used for DeepFlame's dependencies, and the channel needs to be ``conda-forge``. Otherwise, there might be an error regarding shared library, ``libmkl_rt.so.2``.

More detailed instruction of converting mechanisms can be found on `Cantera official website <https://cantera.org/tutorials/ck2yaml-tutorial.html>`_. 


Flame Speed
======================
``flameSpeed.C`` is another utility in DeepFlame. The flame is located at the maximum temperature gradient at a certain time, and its speed is equal to the maximum gradient porpagation speed subtracting the inlet speed.
To use this utility, simply run the commands below after the simulation. 

.. code-block:: bash

    runApplication reconstructPar
    flameSpeed 

A log containing flame thickness, flame location, flame proagation speed, and flame speed at each time step will be presented.

.. Note:: This utility only applies to one-dimensional cases. Similar logs can also exit when it is run for two or three dimensional cases, but results are not physical. 