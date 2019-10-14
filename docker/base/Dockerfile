

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









    
#ENV DJCenvname DJCpath