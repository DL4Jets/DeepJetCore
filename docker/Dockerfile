

FROM cernml4reco/djcbase:latest

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

RUN cd /usr/local/lib/python2.7/dist-packages/tensorflow_core/include/third_party && \
    mkdir gpus && \
    cd gpus && \
    ln -s /usr/local/cuda cuda



#eclipse rse stuff - make extern

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




    
