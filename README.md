# DeepFlame v0.3.0
DeepFlame is a computational fluid dynamics suite for single or multiphase, laminar or turbulent reacting flows at all speeds with machine learning capabilities. It aims to provide an open-source platform bringing together the individual strengths of [OpenFOAM](https://openfoam.org), [Cantera](https://cantera.org) and [pyTorch](https://pytorch.org/) libraries for machine learning assisted reacting flow simulations. It is also has the scope to incorporate next-generation heterogenous supercomputing and AI acceleration infrustructures such as GPU and FPGAs.  

## Dependencies
[OpenFOAM-7](https://openfoam.org/version/7), [Cantera C++ lib 2.6.0](https://anaconda.org/conda-forge/libcantera-devel), [Torch C++ lib 1.11.0](https://pytorch.org/)

## Features
New in v0.3.0:
- 1/2/3D adaptive mesh refinement
- Add Sigma/dynSmag LES turbulence models
- Add functionObjects/field library
- New example reactiveShockTube for dfHighSpeedFoam

New in v0.2.0:
- Dynamic load balancing for chemistry solver

From v0.1.0:
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

## How to install
The installation of DeepFlame is simple and requires [OpenFOAM-7](https://openfoam.org/version/7), [LibCantera](https://anaconda.org/conda-forge/libcantera-devel) and [LibTorch](https://pytorch.org/) . 

### 1. Install [OpenFOAM-7](https://openfoam.org/version/7) (if not already installed)
  
  Quick install (for Ubuntu no later than 20.04): 
```
sudo sh -c "wget -O - https://dl.openfoam.org/gpg.key | apt-key add -"
sudo add-apt-repository http://dl.openfoam.org/ubuntu
sudo apt-get update
sudo apt-get -y install openfoam7
```
  OpenFOAM 7 and ParaView 5.6.0 will be installed in the /opt directory.

### 2. Source your OpenFOAM via the default path below (or your own path for OpenFOAM bashrc)
```
source $HOME/OpenFOAM/OpenFOAM-7/etc/bashrc 
```
### 3. Install [LibCantera](https://anaconda.org/conda-forge/libcantera-devel) via [conda](https://docs.conda.io/en/latest/miniconda.html#linux-installers)
```
conda create -n libcantera

conda activate libcantera

conda install -c cantera libcantera-devel
```
Note: Check your Miniconda3/envs/libcantera directory and make sure the install was successful (lib/ include/ etc. exist).

### 4. Clone the [DeepFlame repository](https://github.com/deepmodeling/deepflame-dev)
```
git clone https://github.com/deepmodeling/deepflame-dev.git

cd deepflame-dev
```
### 5. Install precompiled [LibTorch](https://pytorch.org/) 
```
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcpu.zip

unzip libtorch-cxx11-abi-shared-with-deps-1.11.0+cpu.zip -d thirdParty
```
### 6. Install DeepFlame
```
. install.sh
```
Note: Some compiling issues may happen due to system compatability. Instead of using conda installed Cantera C++ lib and the downloaded Torch C++ lib, try to compile your own Cantera and Torch C++ libraries.

## Running DeepFlame examples
1. Source your OpenFOAM, for example (depends on your OpenFOAM path):
```
source $HOME/OpenFOAM/OpenFOAM-7/etc/bashrc 
```
2. Source deepflame-dev/bashrc, for example (depends on your DeepFlame path):
```
source $HOME/deepflame-dev/bashrc
```
3. Go to an example case directory, for example:
```
cd $HOME/deepflame-dev/examples/df0DFoam/zeroD_cubicReactor/H2/cvodeSolver

./Allrun
```

Note: For the example cases with torchSolver, an additional DNN model file in the `.pt` format is required. Please contact the developers if you would like a test run. 


## Citing DeepFlame
If you use DeepFlame for a publication, please use the citation: 

DeepFlame: A computational fluid dynamics suite for multiphase turbulent reacting flows at all speeds with machine learning. URL:https://github.com/deepmodeling/deepflame-dev, 2022.

## Useful resources
- First release v0.1.0 introduction talk (in Chinese) on [DeepModeling Community's official bilibili website](https://www.bilibili.com/video/BV1Vf4y1f7wB?vd_source=309a67109ca33c4ef79bf506f8ce70ab).
