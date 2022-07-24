#!/usr/bin/bash

BASE_IMAGE_TAG=cu11.6

COMMIT=manual


docker build $FORCE_NO_CACHE -t cernml4reco/deepjetcore3:latest . \
       --build-arg BUILD_DATE="$(date)" --build-arg BASE_IMAGE_TAG=$BASE_IMAGE_TAG \
       --build-arg COMMIT=$COMMIT