FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]

RUN sed -i "s,# deb http://archive.canonical.com/ubuntu,deb http://archive.canonical.com/ubuntu,g" /etc/apt/sources.list
RUN apt-get update 

RUN apt-get install -y python3.10-dev python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools
RUN ldconfig

#basic user tools
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y git wget nano emacs \
    evince eog ffmpeg unzip zsh python3-tk locales \
    htop vim cmake

RUN pip3 install numpy scikit-learn scikit-image \
                 h5py matplotlib uproot3 uproot Pillow \
                 scipy seaborn opencv-python easydict tqdm Cython numba \
                 gpustat setGPU plotly dash awkward hist


# more specifics

RUN pip3 install bayesian-optimization mgzip pyyaml wandb hdbscan tables


# 2.15 has a custom kernel compilation bug that is not being fixed before a patch or something, 2.14 cuda issues...
RUN export DEBIAN_FRONTEND=noninteractive && \
    pip3 --no-cache-dir install tensorflow==2.13 

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


