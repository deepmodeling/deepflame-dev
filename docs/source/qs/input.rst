Brief Introduction to Inputs
======================================
The dictionary ``CanteraTorchProperties`` is the original dictionary of DeepFlame. It reads in network related parameters and configurations. It typically looks like:

.. code-block::

    chemistry           on;
    CanteraMechanismFile "ES80_H2-7-16.yaml";
    transportModel "Mix";
    odeCoeffs
    {
        "relTol"   1e-15;
        "absTol"   1e-24;
    }
    inertSpecie        "N2";
    zeroDReactor
    {
        constantProperty "pressure";
    }

    splittingStretagy false;

    TorchSettings
    {
        torch on;
        GPU   off;
        log  on;
        torchModel "HE04_Hydrogen_ESH2_GMS_sub_20221101"; 
        coresPerNode 4;

    }
    loadbalancing
    {
            active  false;
            //log   true;
    }

In the above example, the meanings of the parameters are:

* ``CanteraMechanismFile``: the name of the reaction mechanism file.
* ``transportModel``: the default model is *Mix*, but other models including *UnityLewis* and *Multi* are also availabile.
* ``constantProperty``: property set to be constant during reaction. It can be set to *pressure* or *volume*.
* ``odeCoeffs``: the ode tolerance. 1e-15 and 1e-24 are used for network training, so they should be kept the same when comparing results with and without DNN. Default values are 1e-9 and 1e-15.
* ``TorchSettings``: all paramenters regarding the usage of DNN. This section will not be read in CVODE cases.
* ``torch``: the switch used to control the on and off of DNN. If users are running CVODE, this needs to be switched off.
* ``GPU``: the switch used to control whether GPU or CPU is used to carry out inference.
* ``torchModel``: name of network.     
* ``coresPerNode``: If you are using one node on a cluster or using your own PC, set this parameter to the actual number of cores used to run the task. If you are using more than one node on a cluster, set this parameter the total number of cores on one node. The number of GPUs used is auto-detected.

The dictionary ``CanteraTorchProperties`` is the original dictionary of DeepFlame. It reads in network related parameters and configurations. It typically looks like:

.. code-block::

    combustionModel  flareFGM;//PaSR,EDC

    EDCCoeffs
    {
        version v2005;
    }

    PaSRCoeffs
    {
       mixingScale
       {
          type   globalScale;//globalScale,kolmogorovScale,geometriMeanScale,dynamicScale 

          globalScaleCoeffs
          {
            Cmix  0.01;
          }
       }
       chemistryScale
       {
          type  formationRate;//formationRate,globalConvertion
          formationRateCoeffs
          {}
       }

    }  
    
    flareFGMCoeffs
    {
      buffer           false;
      scaledPV         false;
      combustion       false;
      ignition         false;
      solveEnthalpy    false;
      flameletT        false;
      relaxation       false;
      DpDt             false;
    /*ignition         false;
      ignBeginTime     0.1;
      ignDurationTime  0.0;
      x0               0.0;
      y0               0.0;
      z0               0.0;
      R0               0.0;*/
      Sct              0.7;
      bufferTime       0.0;
      speciesName      ("CO");
    }

In the above example, the meanings of the parameters are:

* ``combustionModel``: the name of the combustion model, alternative models include PaSR, EDC, flareFGM.
* ``EDCCoeffs, PaSRCoeffs, flareFGMCoeffs``: model cofficients we need to define.
* ``mixingScale``: turbulent mixing time scale including globalScale,kolmogorovScale,geometriMeanScale,dynamicScale.
* ``chemistryScale``: chemistry reaction time scale including formationRate,globalConvertion  .
* ``buffer``: switch for buffer time.
* ``scaledPV``:the switch is used to determine whether to use scaled progress variables or not.
* ``combustion``:the switch is used to control whether the chemical reactions are on or off.
* ``ignition``:the switch is used to control whether the ignition is on or off.     
* ``solveEnthalpy``:the switch is used to determine whether to solve enthalpy equation or not.
* ``flameletT``:the switch is used to determine whether to read flame temperature from table or not.
* ``relaxation``:the switch is used to determine whether to use relaxation iteration for transport equations or not.
* ``DpDt``:the switch is used to determine whether to include material derivatives or not.
* ``ignBeginTime``:beginning time of ignition.
* ``ignDurationTime``:duration time of ignition.
* ``x0, y0, z0``:coordinate of ignition center.
* ``R0``:radius of ignition region.
* ``Sct``:turbulent Schmidt number, default value is set as 0.7.
* ``speciesName``:name of species we need to lookup.
