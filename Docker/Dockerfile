FROM nvidia/cuda:10.2-base-ubuntu18.04

# Install apt dependencies
RUN apt-get update && apt-get install -y \
	git \
	vim \
	wget \
	software-properties-common \
	curl

# Install python3.7 and pip
RUN apt-get install -y python3.7 python3-pip

# install python dependencies
RUN python3.7 -m pip install pip --upgrade
RUN python3.7 -m pip install numpy \
                             scipy \
                             matplotlib \
                             scikit-image \
                             transforms3d \
                             tqdm \
                             opencv-python \
                             cython \
                             open3d==0.10.0 \
                             scikit-learn \
                             scikit-image \
                             pyyaml \
                             pypng \
                             vtk

# Other packages
RUN apt-get install -y freeglut3-dev
