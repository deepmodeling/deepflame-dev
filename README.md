<p align="center">
    <a href="https://github.com/deepmodeling/deepflame-dev">
        <img src="https://user-images.githubusercontent.com/121787251/218061666-eb9e4188-d368-41d0-8ed6-fe5121699efe.jpg">
    </a>
    <a href="https://github.com/deepmodeling/deepflame-dev/releases">
        <img src="https://img.shields.io/github/v/release/deepmodeling/deepflame-dev?include_prereleases&label=latest%20release&rgb(0%2C%20113%2C%20189)">
    </a>    
    <a href="https://github.com/deepmodeling/deepflame-dev/pulls">
        <img src="https://img.shields.io/badge/contributions-welcome-red.svg?color=rgb(48%2C%20185%2C%20237)">
    </a>    
    <a href="https://github.com/deepmodeling/deepflame-dev/blob/master/LICENSE">
        <img src="https://img.shields.io/github/license/deepmodeling/deepflame-dev?logo=GitHub&color=rgb(255%2C%20232%2C%2054)">
    </a>    
    <a href="https://deepflame.deepmodeling.com/en/latest/">
        <img src="https://img.shields.io/website?label=Documentation%20HomePage&up_message=online&url=https%3A%2F%2Fdeepflame.deepmodeling.com%2Fen%2Flatest%2F&color=rgb(241%2C%20155%2C%2068)">
    </a>  
    <a href="https://doi.org/10.1016/j.cpc.2023.108842">
        <img src="https://img.shields.io/badge/DOI-10.1016%2Fj.cpc.2023.108842-black?color=rgb(232%2C%2093%2C%2050)">
    </a> 
</p>

DeepFlame is a deep learning empowered computational fluid dynamics package for single or multiphase, laminar or turbulent, reacting flows at all speeds. It aims to provide an open-source platform to combine the individual strengths of [OpenFOAM](https://openfoam.org), [Cantera](https://cantera.org), and [PyTorch](https://pytorch.org/) libraries for deep learning assisted reacting flow simulations. It also has the scope to leverage the next-generation heterogenous supercomputing and AI acceleration infrastructures such as GPU and FPGA.

The neural network models used in the tutorial examples can be found at– [AIS Square](https://www.aissquare.com/). To run DeepFlame with DNN, download the DNN model [DF-ODENet](https://www.aissquare.com/models/detail?pageType=models&name=DF-ODENet_DNNmodel&id=197) into the case folder you would like to run.

## Documentation
Detailed guide for installation and tutorials is available on [our documentation website](https://deepflame.deepmodeling.com).

## Features
New in v1.3.0 (2023/12/30):
- Complete the full-loop GPU implementation of the `dfLowMachFoam` solver, enabling efficient execution of all computations on GPU
- Introduce `DF-ODENet` model, which utilizes sampling from canonical combustion simulation configurations to reduce training costs and improve computational efficiency
- Support Large Eddy Simulation (LES) and two-phase combustion simulation capabilities
- Expand the `flareFGM` table to six dimensions and add support for neural network replacement of certain physical quantities in the new six-dimensional `flareFGM` table
- Support multi-GPU and multi-processor execution through the `DeepFGM` neural network interface
- Modify Cantera's approach to transport property calculations to support real fluid thermophysical property calculation of multi-component reactive flows and integrate neural networks for updating real fluid thermophysical properties
- Add new example cases and update the documentation homepage to provide more comprehensive installation and usage instructions 

New in v1.2.0 (2023/06/30):
- Enable GPU acceleration for fast and efficient discrete matrix construction for solving partial differential equations
- Introduce `DeePFGM` model: a neural network-based approach to replace the flamelet database of the FGM model and reduce memory requirement
- Support real fluid density calculation with Cantera's PR/RK equation of state and updated the calculation of isentropic compression coefficient (psi)
- Improve dfHighSpeedFoam solver
  - Apply flux splitting to the convective term of the species equations for consistency and accuracy
  - Adopt Strong Stability Preserving Runge-Kutta (RKSSP) time scheme for enhanced stability and performance
- Incorporate [`WENO scheme`](https://github.com/WENO-OF/WENOEXT) and [`libROUNDSchemes`](https://github.com/advanCFD/libROUNDSchemes) as third-party submodules for convective flux reconstruction
- Provide interface to access the reaction rates of each component in a given elementary reaction
- Implement mechanism file detection function to verify the validity of the mechanism file input
- Capture and report Cantera errors for better error handling and user experience

New in v1.1.0 (2023/03/31):
- Add FGM model
- Add GPU-compatible linear solver [AmgX](https://github.com/NVIDIA/AMGX) (adopted from [petsc4Foam](https://develop.openfoam.com/modules/external-solver) and [FOAM2CSR](https://gitlab.hpc.cineca.it/openfoam/foam2csr))
- Add new load balancing algorithm
- Add support for solving chemical source term simultaneously on GPU (DNN) and CPU (CVODE)
- Add support for compilation using CMake
- Improve DNN solving procedure when using pure CPU
- Reconstruct `dfChemistryModel`
- Update chemical and mixing time scale models in PaSR combustion model

New in v1.0.0 (2022/11/15):
- Add support for the parallel computation of DNN using libtorch on multiple GPUs 
- Add TCI model

New in v0.5.0 (2022/10/15):
- Add support for the parallel computation of DNN via single and multiple GPUs
- Add access for utilising PyTorch

New in v0.4.0 (2022/09/26):
- Adapt combustion library from OpenFOAM into DeepFlame
- `laminar`; `EDC`; `PaSR` combustion models

New in v0.3.0 (2022/08/29):
- 1/2/3D adaptive mesh refinement (2/3D adopted from [SOFTX_2018_143](https://github.com/ElsevierSoftwareX/SOFTX_2018_143) and [multiDimAMR](https://github.com/HenningScheufler/multiDimAMR))
- Add Sigma/dynSmag LES turbulence models
- Add functionObjects/field library
- New example reactiveShockTube for `dfHighSpeedFoam`

New in v0.2.0 (2022/07/25):
- Dynamic load balancing for chemistry solver (adopted from [DLBFoam](https://github.com/blttkgl/DLBFoam-1.0))

From v0.1.0 (2022/06/15):
- Native Cantera reader for chemical mechanisms in `.cti`, `.xml` or `.yaml` formats
- Full compatiblity with Cantera's `UnityLewis`, `Mix` and `Multi` transport models
- Zero-dimensional constant pressure or constant volume reactor solver `df0DFoam`
- Pressued-based low-Mach number reacting flow solver `dfLowMachFoam`
- Density-based high-speed reacting flow solver `dfHighSpeedFoam`
- Two-phase Lagrangian/Euler spray reacting flow solver `dfSprayFoam`
- Cantera's native SUNDIALS CVODE solver for chemical reaction rate evaluation
- Torch's tensor operation functionality for neutral network I/O and calculation
- Interface for DNN model to obtain chemical reaction rates
- Multiple example and tutorial cases with `Allrun` and `Allclean` scripts
  - 0D Perfectly Stirred Reactor
  - 1D Freely Propagating Premixed Flame
  - 2D Lifted Partially Premixed Triple Flame
  - 3D Taylor-Green Vortex with Flame
  - 1D Detotation Wave in Homogeneous Premixed Mixture
  - 3D Aachen Bomb Spray Flame


## Useful resources
### DeepModeling Community's official bilibili website: 
- [First release v0.1.0 introduction talk (in Chinese)](https://www.bilibili.com/video/BV1Vf4y1f7wB?vd_source=309a67109ca33c4ef79bf506f8ce70ab)
- Formal release of v1.0.0 introduction talk [(in English)](https://www.bilibili.com/video/BV1jv4y1U7YM/?spm_id_from=333.788&vd_source=309a67109ca33c4ef79bf506f8ce70ab) and [(in Chinese)](https://www.bilibili.com/video/BV14P411u75u/?spm_id_from=333.788&vd_source=309a67109ca33c4ef79bf506f8ce70ab)
