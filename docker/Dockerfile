#FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
#this is the last version without memory leak afaik
FROM tensorflow/tensorflow:2.4.3-gpu 

SHELL ["/bin/bash", "-c"]

# tf compile prep; needs to be adapted to the actual cuda version and the python version!
RUN mkdir -p /usr/local/lib/python3.6/dist-packages/tensorflow/include/third_party/gpus/cuda/include && cp -r /usr/local/cuda/targets/x86_64-linux/include/* /usr/local/lib/python3.6/dist-packages/tensorflow/include/third_party/gpus/cuda/include/


RUN sed -i "s,# deb http://archive.canonical.com/ubuntu,deb http://archive.canonical.com/ubuntu,g" /etc/apt/sources.list
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update 

RUN apt-get install -y python3-pip # python3.11-dev python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools
RUN ldconfig

#basic user tools
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y git wget nano emacs \
    evince eog ffmpeg unzip zsh python3-tk locales \
    htop vim cmake build-essential 

RUN pip3 install numpy scikit-learn scikit-image 
RUN pip3 install h5py matplotlib 
RUN pip3 install uproot3 uproot Pillow 
RUN pip3 install scipy seaborn opencv-python easydict tqdm Cython numba 
RUN pip3 install gpustat setGPU 
RUN pip3 install plotly awkward hist Cython
#RUN pip3 install  dash

# more specifics

RUN pip3 install bayesian-optimization mgzip pyyaml wandb tables
RUN apt-get install -y libpython3.6-dev
#RUN pip3 install --no-binary :all: hdbscan

# 2.15 has a custom kernel compilation bug that is not being fixed before a patch or something, 2.14 cuda issues...
#RUN export DEBIAN_FRONTEND=noninteractive && \
#    pip3 --no-cache-dir install tensorflow==2.13 
RUN apt-get install -y cmake
#strict dependency
RUN python3 -m pip install git+https://github.com/jkiesele/djcdata 

# the upper part can be cached usually, this one needs to be updated
ARG BUILD_DATE
LABEL org.label-schema.build-date=$BUILD_DATE
ARG COMMIT

RUN mkdir -p /usr/share/DJC/
RUN cd /usr/share/DJC/ && git clone https://github.com/DL4Jets/DeepJetCore &&\
    cd DeepJetCore && git checkout $COMMIT 

# nothing to be compiled anymore
    
ENV DEEPJETCORE /usr/share/DJC/DeepJetCore
ENV PYTHONPATH="/usr/share/DJC/DeepJetCore/../:${PYTHONPATH}"
ENV PATH="/usr/share/DJC/DeepJetCore/bin:${PATH}"


