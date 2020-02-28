

FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 


SHELL ["/bin/bash", "-c"]

RUN apt-get update
RUN apt-get install -y python2.7-dev python-pip nano emacs
RUN apt-get install -y wget
RUN apt-get install -y libboost-all-dev
RUN apt-get install -y git


## root part
RUN apt-get install -y libx11-dev libxpm-dev libxft-dev libxext-dev
RUN apt-get install -y cmake

RUN mkdir /temproot && \
    cd /temproot && \
    git clone http://github.com/root-project/root.git && \
    cd root && \
    git checkout -b v6-18-04 v6-18-04 
RUN cd /temproot && \
    mkdir build && \
    cd build && \
    cmake ../root 
RUN cd /temproot/build && \
    make -j5 && \
    make install && \
    rm -rf /temproot


#ensure import ROOT
ENV PYTHONPATH="/usr/local/lib:${PYTHONPATH}"


RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y python-tk
RUN apt-get install -y locales

RUN pip install --upgrade pip

RUN pip install numpy
RUN pip install root_numpy
RUN pip install scikit-learn
RUN pip install scikit-image
RUN pip install h5py
RUN pip install uproot Pillow scipy seaborn
RUN pip install scikit-learn
RUN pip install scikit-image
RUN pip install matplotlib

#may be useful
RUN pip install opencv-python
RUN pip install easydict tqdm
RUN pip install Cython
RUN pip install numba

#already in the container from nvidia
#RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.243-1_amd64.deb &&\
#    dpkg -i cuda-repo-ubuntu1804_10.1.243-1_amd64.deb &&\
#    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub &&\
#    apt-get update &&\
#    wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb &&\
#    apt install -y ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb &&\
#    apt-get update
#    
#RUN export DEBIAN_FRONTEND=noninteractive && \
#    apt-get install -y  --allow-downgrades --allow-change-held-packages --no-install-recommends \
#    cuda-10-1 \
#    libcudnn7=7.6.4.38-1+cuda10.1  \
#    libcudnn7-dev=7.6.4.38-1+cuda10.1
#    
##will also get new cuda version
#RUN export DEBIAN_FRONTEND=noninteractive && \
#    apt-get install -y --no-install-recommends libnvinfer6=6.0.1-1+cuda10.1 \
#    libnvinfer-dev=6.0.1-1+cuda10.1 \
#    libnvinfer-plugin6=6.0.1-1+cuda10.1

RUN export DEBIAN_FRONTEND=noninteractive && \
    pip --no-cache-dir install tensorflow-gpu==2.1

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update --fix-missing 
    
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y krb5-user && \
    sed -i 's/default_realm = ATHENA.MIT.EDU/default_realm = CERN.CH/g' /etc/krb5.conf 
    
#for me ;)    

RUN sed -i "s,# deb http://archive.canonical.com/ubuntu,deb http://archive.canonical.com/ubuntu,g" /etc/apt/sources.list

RUN apt-get update --fix-missing 
RUN apt-get install -y zsh 

RUN apt-get install -y ffmpeg
#not yet RUN apt-get install -y nvtop

##gpu specific stuff
RUN pip install gpustat 
RUN pip install setGPU


RUN apt-get install -y unzip


RUN sed -i "s,# deb http://archive.canonical.com/ubuntu,deb http://archive.canonical.com/ubuntu,g" /etc/apt/sources.list
RUN apt update
# do not upgrade all, because of cudnn versions etc!!!
#for eclipse stuff
RUN apt install -y default-jre 
RUN apt install -y openjdk-11-jre-headless 
RUN apt install -y openjdk-8-jre-headless 
RUN apt install -y default-jdk
#RUN apt install -y openjdk-9-jre-headless 



## torch/jupyter stuff 
RUN pip install torch future

RUN pip install gensim && \
    pip install jupyter && \
    pip install prompt-toolkit && \
    pip install seaborn 
    
EXPOSE 8888


RUN cd /usr/local/lib/python2.7/dist-packages/tensorflow_core/include/third_party && \
    mkdir gpus && \
    cd gpus && \
    ln -s /usr/local/cuda cuda
    
################################################################################
# the actual DeepJetCore. Make sure this gets built every time

ARG BUILD_DATE
LABEL org.label.build-date=$BUILD_DATE

ENV DEEPJETCORE /usr/share/DJC/DeepJetCore
RUN ldconfig
RUN cd /usr/share && \
    mkdir DJC && \
    cd DJC && \
    git clone https://github.com/DL4Jets/DeepJetCore && \
    cd DeepJetCore && \
    source docker_env.sh && \
    cd compiled && \
    make -j4


ENV PATH="/usr/share/DJC/DeepJetCore/bin:${PATH}"

#/usr/local/lib is for root
ENV PYTHONPATH="/usr/share/DJC/DeepJetCore/../:${PYTHONPATH}"
ENV LD_LIBRARY_PATH="/usr/share/DJC/DeepJetCore/compiled:${LD_LIBRARY_PATH}"


# helpers for ragged and cuda compilation



# The fix for TensorFlow

    