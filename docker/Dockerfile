
FROM cernml4reco/djcbase:py2

RUN export DEBIAN_FRONTEND=noninteractive && \
    pip --no-cache-dir install tensorflow-gpu==1.15
    
RUN cd /usr/local/lib/python2.7/dist-packages/tensorflow_core/include/third_party && \
    mkdir gpus && \
    cd gpus && \
    ln -s /usr/local/cuda cuda
    
## torch/jupyter stuff 
RUN pip install torch future

RUN pip install gensim && \
    pip install jupyter && \
    pip install prompt-toolkit && \
    pip install seaborn 
    
EXPOSE 8888


    
################################################################################
# the actual DeepJetCore. Make sure this gets built every time

ARG BUILD_DATE
LABEL org.label-schema.build-date=$BUILD_DATE

ENV DEEPJETCORE /usr/share/DJC/DeepJetCore
RUN ldconfig
RUN cd /usr/share && \
    mkdir DJC && \
    cd DJC && \
    git clone https://github.com/DL4Jets/DeepJetCore && \
    cd DeepJetCore && \
    git checkout 2.2_tf1.15 &&\
    source docker_env.sh && \
    cd compiled && \
    make -j4


ENV PATH="/usr/share/DJC/DeepJetCore/bin:${PATH}"

#/usr/local/lib is for root
ENV PYTHONPATH="/usr/share/DJC/DeepJetCore/../:${PYTHONPATH}"
ENV LD_LIBRARY_PATH="/usr/share/DJC/DeepJetCore/compiled:${LD_LIBRARY_PATH}"


# helpers for ragged and cuda compilation



# The fix for TensorFlow

    