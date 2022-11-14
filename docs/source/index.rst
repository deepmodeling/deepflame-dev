.. Testing documentation master file, created by
   sphinx-quickstart on Thu Aug 25 16:15:41 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=======================================
Welcome to DeepFlame's Documentation
=======================================

.. _Overview:

DeepFlame is a deep learning empowered computational fluid dynamics package for single or multiphase, laminar or turbulent, reacting flows at all speeds. It aims to provide an open-source platform to combine the individual strengths of `OpenFOAM <https://openfoam.org/>`_, `Cantera <https://cantera.org/>`_ and `PyTorch <https://pytorch.org/libraries>`_ libraries for deep learning assisted reacting flow simulations. It is also has the scope to incorporate next-generation heterogenous supercomputing and AI acceleration infrustructures such as GPU and FPGA.

The deep learning algorithms and models used in the DeepFlame tutorial examples are developed and trained independently by our collaborators team – `DeepCombustion <https://github.com/deepcombustion/deepcombustion/>`_. Please refer to their website for detailed information.

.. Note:: This project is under active development.  
   

.. _Installation:

.. toctree::
   :maxdepth: 3
   :numbered:
   :caption: Quick Start
   :glob:

   qs/install
   qs/examples
   qs/input
   

.. _solvers:

.. toctree::
   :maxdepth: 3
   :numbered:
   :caption: Solvers
   :glob:

   solvers/df0DFoam
   solvers/dfLowMachFoam
   solvers/dfHighSpeedFoam
   solvers/dfSprayFoam


.. _Utility:

.. toctree::
   :maxdepth: 3
   :numbered:
   :caption: Utility
   :glob:

   utility

.. _project-details:

.. toctree::
   :maxdepth: 3
   :numbered:
   :caption: Project Details

   
   contributors
   citation
   gpl

.. _contributing:

.. toctree::
   :maxdepth: 3
   :caption: Contributing to DeepFlame


   pr




