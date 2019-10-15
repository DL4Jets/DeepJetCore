

FROM cernml4reco/djc_ubuntu

#needs to run first: nvidia-modprobe -u -c=0 outside container

RUN pip install gpustat 
RUN pip install setGPU
#this one is big
RUN pip --no-cache-dir install tensorflow-gpu==1.14

