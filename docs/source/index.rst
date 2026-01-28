.. Testing documentation master file, created by
   sphinx-quickstart on Thu Aug 25 16:15:41 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=======================================
Welcome to DeepFlame's Documentation
=======================================

.. _Overview:

DeepFlame is a deep learning empowered computational fluid dynamics package for single or multiphase, laminar or turbulent, reacting flows at all speeds. It aims to provide an open-source platform to combine the individual strengths of `OpenFOAM <https://openfoam.org/>`_, `Cantera <https://cantera.org/>`_, and `PyTorch <https://pytorch.org/libraries>`_ libraries for deep learning assisted reacting flow simulations. It also has the scope to incorporate next-generation heterogenous supercomputing and AI acceleration infrastructures such as GPU and FPGA.

The deep learning algorithms and models used in the DeepFlame tutorial examples are made available in AIS Square for community data sharing – `DF-ODENet <https://www.aissquare.com/models/detail?pageType=models&name=DF-ODENet_DNNmodel&id=197/>`_. Please refer to the website for detailed information.

.. Note:: This project is under active development.  
   

.. _Installation:

.. toctree::
   :maxdepth: 3
   :numbered:
   :caption: Quick Start
   :glob:

   qs/install
   qs/dnn_models
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




