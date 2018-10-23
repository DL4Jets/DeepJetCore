LABEL version="0.0.1" maintainer "Swapneel Mehta <swapneel.mehta@djsce.edu.in>"

FROM cern/cc7-base
RUN yum install -y \
wget \
git \
root \
bzip2
RUN mkdir /home/deepjet-framework && cd /home/deepjet-framework
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN git clone https://github.com/DL4Jets/DeepJet \
&& git clone https://github.com/DL4Jets/DeepJetCore

