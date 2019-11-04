

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

RUN pip --no-cache-dir install tensorflow==1.15


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
    
    
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y krb5-user && \
    sed -i 's/default_realm = ATHENA.MIT.EDU/default_realm = CERN.CH/g' /etc/krb5.conf 
    
#for me ;)    
RUN apt-get install -y zsh 

##gpu specific stuff
RUN pip install gpustat 
RUN pip install setGPU





    
#ENV DJCenvname DJCpath