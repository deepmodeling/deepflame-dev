<p align="center">
    <a href="https://github.com/deepmodeling/deepflame-dev">
        <img src="https://user-images.githubusercontent.com/121787251/218061666-eb9e4188-d368-41d0-8ed6-fe5121699efe.jpg">
    </a>
    <a href="https://github.com/deepmodeling/deepflame-dev/releases">
        <img src="https://img.shields.io/github/v/release/deepmodeling/deepflame-dev?include_prereleases&label=latest%20release&style=for-the-badge">
    </a>    
    <a href="https://github.com/deepmodeling/deepflame-dev/pulls">
        <img src="https://img.shields.io/badge/contributions-welcome-red.svg?style=for-the-badge">
    </a>    
    <a href="https://github.com/deepmodeling/deepflame-dev/blob/master/LICENSE">
        <img src="https://img.shields.io/github/license/deepmodeling/deepflame-dev?color=yellow&logo=GitHub&style=for-the-badge">
    </a>    
    <a href="https://deepflame.deepmodeling.com/en/latest/">
        <img src="https://img.shields.io/website?label=Documentation%20HomePage&style=for-the-badge&up_message=online&url=https%3A%2F%2Fdeepflame.deepmodeling.com%2Fen%2Flatest%2F">
    </a>  
</p>

DeepFlame is a deep learning empowered computational fluid dynamics package for single or multiphase, laminar or turbulent, reacting flows at all speeds. It aims to provide an open-source platform to combine the individual strengths of [OpenFOAM](https://openfoam.org), [Cantera](https://cantera.org), and [PyTorch](https://pytorch.org/) libraries for deep learning assisted reacting flow simulations. It also has the scope to leverage the next-generation heterogenous supercomputing and AI acceleration infrastructures such as GPU and FPGA.

The deep learning algorithms and models used in the DeepFlame tutorial examples are developed and trained independently by our collaborators team – [DeepCombustion](https://github.com/deepcombustion/deepcombustion). Please refer to their website for detailed information.

## Documentation
Detailed guide for installation and tutorials is available on [our documentation website](https://deepflame.deepmodeling.com).

## Features
New features as of 2023/03/01:
- New load balancing algorithm
- Add support for solving chemical source term simultaneously on GPU (DNN) and CPU (CVODE)
- Add FGM model
- Reconstruct dfChemistryModel 

New in v1.0.0 (2022/11/15):
- Add support for the parallel computation of DNN using libtorch on multiple GPUs 
- Add TCI model

New in v0.5.0 (2022/10/15):
- Add support for the parallel computation of DNN via single and multiple GPUs
- Add access for utilising PyTorch

New in v0.4.0 (2022/09/26):
- Adapt combustion library from OpenFOAM into DeepFlame
- laminar; EDC; PaSR combustion models

New in v0.3.0 (2022/08/29):
- 1/2/3D adaptive mesh refinement (2/3D adopted from [SOFTX_2018_143](https://github.com/ElsevierSoftwareX/SOFTX_2018_143) and [multiDimAMR](https://github.com/HenningScheufler/multiDimAMR))
- Add Sigma/dynSmag LES turbulence models
- Add functionObjects/field library
- New example reactiveShockTube for dfHighSpeedFoam

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
