FROM ubuntu:20.04
SHELL ["/bin/bash", "-c"]
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y --no-install-recommends \
&& apt-get update \
	&& apt-get install -y \
		sudo \
		wget \
        zip\
        unzip\
        git\
        software-properties-common

RUN sudo sh -c "wget -O - https://dl.openfoam.org/gpg.key | apt-key add -" \
&& sudo add-apt-repository http://dl.openfoam.org/ubuntu \
&& DEBIAN_FRONTEND=noninteractive \
&& sudo apt-get update
RUN DEBIAN_FRONTEND=noninteractive\
sudo apt-get -y install openfoam7 \
&& sudo apt-get install --only-upgrade openfoam7 \
#RUN source opt/openfoam7/etc/bashrc 
&& source /opt/openfoam7/etc/bashrc



RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcpu.zip\
&& unzip libtorch-cxx11-abi-shared-with-deps-1.11.0+cpu.zip -d thirdParty \
&& rm libtorch-cxx11-abi-shared-with-deps-1.11.0+cpu.zip 
#RUN . ~/miniconda3/etc/profile.d/conda.sh 
#SHELL ["/bin/bash", "-c"] 
#RUN /bin/bash -c "source activate work"
#ENV PATH="/root/miniconda3/bin:$PATH"
#RUN /source activate"
#RUN conda create -n libcantera 
#RUN conda init --all 
#SHELL ["/bin/bash", "-c"] 
#RUN /bin/bash -c "source .bashrc" 
#RUN . ~/miniconda3/etc/profile.d/conda.sh 


#RUN conda activate libcantera

#RUN conda update -n base -c defaults conda &&\
#conda activate libcantera

#RUN conda activate libcantera

#RUN conda install -c cantera libcantera-devel

RUN git clone --depth 1 https://github.com/JX278/deepflame-dev.git 

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh\
&& sh Miniconda3-latest-Linux-x86_64.sh -b\
&& source ~/miniconda3/etc/profile.d/conda.sh \
&& conda create -n libcantera \
&& conda activate libcantera \
&& conda install -c cantera libcantera-devel \
&& rm Miniconda3-latest-Linux-x86_64.sh \
&& cd deepflame-dev \
&& source ~/miniconda3/etc/profile.d/conda.sh \
&& source /opt/openfoam7/etc/bashrc \
&& . install.sh 
