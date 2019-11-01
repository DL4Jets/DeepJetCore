#!/bin/bash


tag=$1

cd base
docker build -t "cernml4reco/djc_ubuntu_rootbase:${tag}" .
cd ../DJCEnv
docker build -t "cernml4reco/djc_ubuntu:${tag}" .
cd ../gpuAdd
docker build -t "cernml4reco/djc_ubuntu_gpu:${tag}" .
cd ..
