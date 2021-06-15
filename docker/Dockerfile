
FROM cernml4reco/djcbase:cu11.1




################################################################################
# Tensorflow

RUN export DEBIAN_FRONTEND=noninteractive && \
    pip3 --no-cache-dir install tensorflow-gpu==2.4.0 tensorflow_probability
    


RUN cd /usr/local/lib/python3.6/dist-packages/tensorflow/include/third_party && \
    mkdir gpus && \
    cd gpus && \
    ln -s /usr/local/cuda cuda    
    
RUN ln -s /usr/local/cuda-11.1/lib64/libcusolver.so.11 /usr/local/cuda-11.1/lib64/libcusolver.so.10
    
    
################################################################################
# the actual DeepJetCore. Make sure this gets built every time

ARG BUILD_DATE
LABEL org.label-schema.build-date=$BUILD_DATE

ENV DEEPJETCORE /usr/share/DJC/DeepJetCore

RUN ldconfig && \
    cd /usr/share && \
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

# for kenneth
RUN apt-get install -y vim

# QKeras

RUN git clone https://github.com/google/qkeras &&\
    cd qkeras &&\
    python3 setup.py build &&\
    python3 setup.py install 
    

    