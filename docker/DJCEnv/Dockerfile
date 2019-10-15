

FROM cernml4reco/djc_ubuntu_rootbase


#
# All additional ML packages and DeepJetCore here
# 
#

SHELL ["/bin/bash", "-c"]

RUN apt-get install -y locales

RUN pip install numpy
RUN pip install root_numpy
RUN pip install scikit-learn
RUN pip install scikit-image
RUN pip install h5py
RUN pip install uproot

#RUN pip install gpustat
#RUN pip install tensorflow-gpu==1.13.1

RUN pip install tensorflow==1.13.1
## RUN pip install Keras==2.0.5

# DEEPJETCORE specifics

ENV DEEPJETCORE /usr/share/DJC/DeepJetCore
RUN ldconfig
RUN cd /usr/share && \
    mkdir DJC && \
    cd DJC && \
    git clone https://github.com/DL4Jets/DeepJetCore && \
    cd DeepJetCore && \
    git checkout docker && \
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

##eos and CERN stuff, ask clemens









    
#ENV DJCenvname DJCpath