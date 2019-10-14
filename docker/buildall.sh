#!/bin/bash

cd base
docker build -t cernml4reco/djc_ubuntu_rootbase:latest .
cd ../DJCEnv
docker build -t cernml4reco/djc_ubuntu:latest .
cd ../gpuAdd
docker build -t cernml4reco/djc_ubuntu_gpu:latest .
cd ..
