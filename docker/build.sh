#!/usr/bin/bash

BASE_IMAGE_TAG=cu11.6

docker build -t cernml4reco/deepjetcore3:latest . \
       --build-arg BUILD_DATE="$(date)"  --build-arg BASE_IMAGE_TAG=$BASE_IMAGE_TAG