Brief Introduction to Inputs
======================================
The dictionary ``CanteraTorchProperties`` is the original dictionay of DeepFlame. It read in netowrk realted parameters and configurations. It typically looks like:

.. code-block::

    chemistry           on;
    CanteraMechanismFile "ES80_H2-7-16.yaml";
    transportModel "Mix";//"UnityLewis";
    odeCoeffs
    {
        //"relTol"   1e-15;
        //"absTol"   1e-24;
    }
    inertSpecie        "N2";

    zeroDReactor
    {
        constantProperty "pressure";
    }

    torch on;
    GPU on;
    torchModel1 "ESH2-sub1.pt";
    torchModel2 "ESH2-sub2.pt";
    torchModel3 "ESH2-sub3.pt";

    torchParameters1
    {
        Tact 700  ;
        Qdotact  3e7;
        coresPerGPU 4;
    }
    torchParameters2
    {
        Tact 2000;
        Qdotact  3e7;
    }
    torchParameters3
    {
        Tact 2000;
        Qdotact  7e8;
    }
    loadbalancing
    {
            active  false;
            //log   true;
    }


In the above example, the meanings of the parameters are:

* ``CanteraMechanismFile``: the name of the reaction mechanism file 
* ``odeCoeffs``: the ode torlerance. 1e-15 and 1e-24 are used for network training, so it should keep the same when comparing results with nd without DNN.
* ``torch``: the switch used to control the on and off of DNN. If users are running CVODE, this needs to be switched off.
* ``GPU``: the switch used to control whether GPU or CPU is used to carry out inference.
* ``torchModel``: name of network.
* ``torchParameters``: thresholds used to decide when to use network.
* ``coresPerGPU``: number of CPU cores on one node.
