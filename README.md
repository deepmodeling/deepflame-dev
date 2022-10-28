# DeepFlame v0.5.0
DeepFlame is a computational fluid dynamics suite for single or multiphase, laminar or turbulent reacting flows at all speeds with machine learning capabilities. It aims to provide an open-source platform bringing together the individual strengths of [OpenFOAM](https://openfoam.org), [Cantera](https://cantera.org) and [Torch](https://pytorch.org/) libraries for deep learning assisted reacting flow simulations. It is also has the scope to incorporate next-generation heterogenous supercomputing and AI acceleration infrustructures such as GPU and FPGAs.

## Dependencies
[OpenFOAM-7](https://openfoam.org/version/7), [Cantera C++ lib 2.6.0](https://anaconda.org/conda-forge/libcantera-devel), [Torch lib 1.11.0](https://pytorch.org/)

## Features
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

### 3. Clone the [DeepFlame repository](https://github.com/deepmodeling/deepflame-dev)
```
git clone https://github.com/deepmodeling/deepflame-dev.git

cd deepflame-dev
```
### 4. Install dependencies and DeepFlame based on your need
DeepFlame supports three compilation choices: no torch, LibTorch, and PyTorch. 
>**Note**: You are encouaged to try all three options, but remember to install the next version in a new terminal to clean previous environment variables.
#### 4.1 No Torch version 
If your are using DeepFlame's cvODE solver without DNN model, just install [LibCantera](https://anaconda.org/conda-forge/libcantera-devel) via [conda](https://docs.conda.io/en/latest/miniconda.html#linux-installers).
```
conda create -n df-notorch
conda activate df-notorch
conda install -c cantera libcantera-devel
```
>**Note**: Check your Miniconda3/envs/libcantera directory and make sure the install was successful (lib/ include/ etc. exist).

If the conda env `df-notorch` is activated, install DeepFlame by running: 

```
. install.sh 
```
If `df-notorch` not activated (or you have a self-complied libcantera), specify the path to your libcantera:
```
. install.sh --libcantera_dir /your/path/to/libcantera/
```

#### 4.2 LibTorch version
If you choose to use LibTorch (C++ API for Torch), first create the conda env and install [LibCantera](https://anaconda.org/conda-forge/libcantera-devel):
```
conda create -n df-libtorch
conda activate df-libtorch
conda install -c cantera libcantera-devel
```
Then you can either install DeepFlame with autodownloaded LibTorch
```
. install.sh --libtorch_autodownload
```
or you can pass your own libtorch path to DeepFlame.

```
. install.sh --libtorch_dir /path/to/libtorch/
```

#### 4.3 PyTorch version
PyTorch version aims to support Python-based progamming for DeepFlame. Run the following commands to install:
```
conda create -n df-pytorch python=3.8
conda activate df-pytorch
conda install -c cantera libcantera-devel
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge 
conda install pybind11

. install.sh --use_pytorch
```
>**Note**: You may come accross an error regarding shared library *libmkl_rt.so.2* when libcantera is installed through cantera channel. If so, go to your conda environment and check the existance of *libmkl_rt.so.2* and *libmkl_rt.so.1*, and then link *libmkl_rt.so.2* to *libmkl_rt.so.1*.
```
cd ~/miniconda3/envs/df-pytorch/lib
ln -s libmkl_rt.so.1 libmkl_rt.so.2
```

> **Note**: Some compiling issues may happen due to system compatability. Instead of using conda installed Cantera C++ lib and the downloaded Torch C++ lib, try to compile your own Cantera and Torch C++ libraries.

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

>**Note**: For the example cases with torchSolver, an additional DNN model file in the `.pt` format is required. Please contact the developers if you would like a test run.



## Citing DeepFlame
**If you use DeepFlame for a publication, please use the citation:**

Runze Mao, Minqi Lin, Yan Zhang, Tianhan Zhang, Zhi-Qin John Xu, Zhi X. Chen. DeepFlame: A deep learning empowered open-source platform for reacting flow simulations (2022). [doi:10.48550/arXiv.2210.07094](https://doi.org/10.48550/arXiv.2210.07094)

**If you have used the DNN model provided from us, please use the citation:**

Tianhan Zhang, Yuxiao Yi, Yifan Xu, Zhi X. Chen, Yaoyu Zhang, Weinan E, Zhi-Qin John Xu. A multi-scale sampling method for accurate and robust deep neural network to predict combustion chemical kinetics. Combust. Flame 245:112319 (2022). [doi:10.1016/j.combustflame.2022.112319](https://doi.org/doi:10.1016/j.combustflame.2022.112319)

## Useful resources
- First release v0.1.0 introduction talk (in Chinese) on [DeepModeling Community's official bilibili website](https://www.bilibili.com/video/BV1Vf4y1f7wB?vd_source=309a67109ca33c4ef79bf506f8ce70ab).
