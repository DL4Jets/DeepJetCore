ARG BASE_IMAGE_TAG
FROM cernml4reco/djcbase:$BASE_IMAGE_TAG

#update if needed
RUN pip3 install --upgrade pip
RUN apt-get update  --fix-missing

########### torch stuff: pretty slow so do that first

#moved from base
RUN pip3 install future gensim jupyter prompt-toolkit
EXPOSE 8888  

# removed torch (geometric) for now, as nobody seems to use it
# RUN pip3 install torch==1.10.2
# RUN pip3 install install torch-scatter torch-sparse torch-cluster \
#                          torch-spline-conv torch-geometric \
#                          -f https://data.pyg.org/whl/torch-1.10.2+cu111.html
# 


RUN apt-get install -y vim

RUN pip3 install bayesian-optimization mgzip mysql-connector-python pyjet pyyaml

################################################################################
# Tensorflow

# TF needs gcc7 for custom ops
# RUN apt-get install -y gcc-7 g++-7 && \
#     update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 7 && \
#     update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 7
RUN g++ --version && gcc --version
    

RUN export DEBIAN_FRONTEND=noninteractive && \
    pip3 --no-cache-dir install tensorflow-gpu==2.4.4  
    
RUN cd /usr/local/lib/python3.6/dist-packages/tensorflow/include/third_party && \
    mkdir gpus && \
    cd gpus && \
    ln -s /usr/local/cuda-11.1 cuda   

#fix for TF 2.4.0
RUN ln -s /usr/local/cuda-11.1/lib64/libcusolver.so.11 /usr/local/cuda-11.1/lib64/libcusolver.so.10
ENV LD_LIBRARY_PATH="/usr/local/cuda-11.1/lib64:${LD_LIBRARY_PATH}"
    

############# some more useful packages, mostly requested by users



################################################################################
# the actual DeepJetCore. Make sure this gets built every time

# prepare the environment (only starting py >3.6)
# RUN cd /usr/lib/x86_64-linux-gnu &&\
#    ln -s libboost_numpy38.so libboost_numpy3.so && \
#    ln -s libboost_python38.so libboost_python3.so


ARG BUILD_DATE
LABEL org.label-schema.build-date=$BUILD_DATE
ARG COMMIT
LABEL djc.commit=$COMMIT

ENV DEEPJETCORE /usr/share/DJC/DeepJetCore

RUN ldconfig && \
    cd /usr/share && \
    mkdir DJC && \
    cd DJC && \
    git clone https://github.com/DL4Jets/DeepJetCore && \
    cd DeepJetCore  && git checkout $COMMIT 
    
    
RUN cd $DEEPJETCORE &&\
    source docker_env.sh && \
    cd compiled && \
    make 


ENV PYTHONPATH="/usr/share/DJC/DeepJetCore/../:${PYTHONPATH}"
ENV LD_LIBRARY_PATH="/usr/share/DJC/DeepJetCore/compiled:${LD_LIBRARY_PATH}"
ENV PATH="/usr/share/DJC/DeepJetCore/bin:${PATH}"


RUN cd /usr/share/DJC/DeepJetCore/testing/unit && \
    python3 test.py
   
    
