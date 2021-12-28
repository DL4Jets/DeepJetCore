
FROM cernml4reco/djcbase:cu11.1


RUN apt-get update --fix-missing


################################################################################
# Tensorflow

RUN export DEBIAN_FRONTEND=noninteractive && \
    pip3 --no-cache-dir install tensorflow-gpu==2.4.0 tensorflow_probability 
    


RUN cd /usr/local/lib/python3.6/dist-packages/tensorflow/include/third_party && \
    mkdir gpus && \
    cd gpus && \
    ln -s /usr/local/cuda cuda    
    
RUN ln -s /usr/local/cuda-11.1/lib64/libcusolver.so.11 /usr/local/cuda-11.1/lib64/libcusolver.so.10
ENV LD_LIBRARY_PATH="/usr/local/cuda-11.1/lib64:${LD_LIBRARY_PATH}"
    



# helpers for ragged and cuda compilation

# for kenneth
RUN apt-get install -y vim

# QKeras

RUN git clone https://github.com/google/qkeras &&\
    cd qkeras &&\
    python3 setup.py build &&\
    python3 setup.py install 
      

#some more useful packages
RUN pip3 install bayesian-optimization mgzip mysql-connector-python


# HLS4ML
RUN pip3 install git+https://github.com/fastmachinelearning/hls4ml.git#egg=hls4ml[profiling]  tensorflow_datasets
RUN pip3 install git+https://github.com/thesps/conifer.git


#add some kerberos for convenience
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y krb5-user 
    
#use the standard cern config
COPY krb5.conf /etc/krb5.conf

RUN apt-get update --fix-missing

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


ENV PYTHONPATH="/usr/share/DJC/DeepJetCore/../:${PYTHONPATH}"
ENV LD_LIBRARY_PATH="/usr/share/DJC/DeepJetCore/compiled:${LD_LIBRARY_PATH}"
ENV PATH="/usr/share/DJC/DeepJetCore/bin:${PATH}"

#run tests
RUN testEnvironment.py

RUN cd /usr/share/DJC/DeepJetCore/testing/unit && \
    python3 test.py
    



    