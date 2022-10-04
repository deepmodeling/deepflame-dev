dfHighSpeedFoam
==================

One-Dimensional Reactive Shock Tube
----------------------------------------


**Problem Description**


The case simulates supersonic inlet flow hitting the wall and then reflected to ignite the premixed gas. The reactive wave will catch the reflected shock wave. This case can also verify the accuracy of our solver in capturing the interaction of convection and reaction.


.. list-table:: Operating Conditions in Brief
   :widths: 40 40 
   :header-rows: 0

   * - Chamber size (x)
     - 0.12m
   * - Initial Gas Density
     - 0.072 kg/m^3 (x<=0.06 m), 0.18075 kg/m^3 (x>0.06 m) 
   * - Initial Gas Pressure
     - 7173 Pa (x<=0.06 m), 35594 Pa (x>0.06 m)
   * - Initial Gas Velocity
     - 0 m/s (x<=0.06 m), -487.34 m/s (x>0.06 m)
   * - Ideal Gas Composition (mole fraction)
     - H2/O2/Ar = 2/1/7 


**Output** 


.. figure:: 1D_reactive_shock_tube.png


   Result of one-dimensional reactive shock tube



One-Dimensional H2/Air Detonation
--------------------------------------------

**Problem Description**


Detonation propagation contains a complex interaction of the leading shock wave and auto-igniting reaction, showing the coupling of shock wave and chemical reaction. This case aims to validate the accuracy of this solver in capturing this process and the propagation speed.


.. list-table:: Operating Conditions in Brief
   :widths: 40 40 
   :header-rows: 0

   * - Chamber size (x)
     - 0.5m
   * - Initial Gas Pressure
     - 90 atm (hot spot), 1 atm (other area)
   * - Initial Gas Temperature
     - 2000 K (hot spot), 300 K  (other area)
   * - Ideal Gas Composition (mole fraction)
     - H2/O2/N2 = 2/1/3.76
       (homogeneous stoichiometric mixture)





**Output** 


.. figure:: 1D_air_detonation.png

   Result of one-dimensional H2/air detonation



Reference
---------------
E S Oran, T. R. Young, J. P. Boris, A. Cohen, Weak and strong ignition. i. Numerical simulations of shock tube experiments, Combustion and Flame 48 (1982) 135-148.