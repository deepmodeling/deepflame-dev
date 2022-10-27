dfLowMachFoam
====================

One-Dimensional Planar Flame
----------------------------------------


**Problem Description**


The case simulates the steady-state 1D freely-propagating flame. The results are able to catch the flame thickness, laminar fame speed and the detailed 1D flame structure. This case demonstrate that the convection-diffusion-reaction algorithms implemented in our solver are stable and accurate.


.. list-table:: Operating Conditions in Brief
   :widths: 40 40 
   :header-rows: 0

   * - Computational Domain length
     - 0.06 m
   * - Mixture
     - Hydrogen-Air
   * - Equivalence Ratio
     - 1.0
   * - Inlet Gas Temperature
     - 300 K


**Output** 


.. figure:: 1D_planar_flame.png
   :width: 500
   :align: center


   Numerical setup of one-dimensional premixed flame and the detailed flame structure obtained by our solver 


Two-Dimensional Jet Flame
--------------------------------------------

**Problem Description**

This case simulates the evolution of a 2D non-premixed planar jet flame to validate the capability of our solver for multi-dimensional applications.

.. list-table:: Operating Conditions in Brief
   :widths: 40 40 
   :header-rows: 0

   * - Computational Domain size (x)
     - 0.03 m * 0.05 m
   * - Jet Composition
     - H2/H2= 1/3 (fuel jet), Air (co-flow)
   * - Initial Velocity   
     - 5 m/s (fuel jet), 1 m/s (co-flow)
   * - Initial Gas Temperature
     - 1400 K (ignition region), 300 K  (other area)



**Output** 

.. figure:: 2D_triple_flame.png
   :width: 500
   :align: center

   Simulation results of the two-dimensional jet flame. 

The initial condition and the evolution of the jet flame are presented in this figure. 

Three-Dimensional reactive Taylor-Green Vortex
---------------------------------------------------

3D reactive Taylor-Green Vortex (TGV) which is a newly established benchmark case for reacting flow DNS codes is simulated here to evaluate the computational performance of our solver. 

**Output** 

The initial and the developed TGV are displayed in the figures below. 

.. figure:: 3D_TGV_initial.png
   :width: 500
   :align: center

   Initial contours and profiles of vorticity magnitude, temperature, and species mass fraction for the reactive TGV

.. figure:: 3D_TGV_0.5ms.png
   :width: 500
   :align: center

   Contours and profiles of temperature and species mass fraction at t = 0.5 ms

**Reference**

A.Abdelsamie, G.Lartigue, C.E.Frouzakis, D.Thevenin The taylor-green vortex as a benchmark for high-fidelity combustion simulations using low-mach solvers, Computers & Fluids 223.