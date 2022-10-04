dfSprayFoam
=================

aachenBomb
------------------

**Problem Description**


This case simulates combustion inside a constant volume chamber based on an experimental setup at RWTH Aachen University. It can mimic, for example, the beginning of power stroke in a four-stroke diesel engine. 



.. list-table:: Operating Conditions in Brief
   :widths: 40 40 
   :header-rows: 0

   * - Chamber size (xyz)
     - 0.02×0.1×0.02m^3
   * - Initial Gas Temperature
     - 800K
   * - Initial Gas Pressure
     - 5MPa
   * - Initial Gas Composition (mass fraction)
     - 23.4% O2, 76.6% N2
   * - Fuel
     - n-heptane
   * - Fuel Temperature at the Nozzle
     - 320K
   * - Fuel Injection Duration
     - 1.25ms
   * - Total Injection Mass
     - 6mg



**Configurations Different from OpenFOAM Case**


Cantera is used instead of the built-in modules of OpenFOAM to solve the chemical reactions. Therefore, a chemical mechanism file in YAML format is required in the case directory, and the full name of the mechanism file ("xxx. YAML") should be the entry after the keyword **CanteraMechanismFile** in *constant/CanteraTorchProperties*. Non-reacting simulation can be conducted by switching the entry after the keyword **chemistry** from **on** to **off** in *constant/CanteraTorchProperties*.