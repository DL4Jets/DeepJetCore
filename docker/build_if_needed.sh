#!/bin/bash

BASE_IMAGE_TAG=cu11.6

git fetch
if [ $(git rev-parse HEAD) != $(git rev-parse @{u}) ]
then
  
  OLD_BASE_ID=$(git rev-parse HEAD:docker/Dockerfile_base) 
  git pull
  NEW_BASE_ID=$(git rev-parse HEAD:docker/Dockerfile_base) 
  
  if [ $OLD_BASE_ID != $NEW_BASE_ID ]
  then
    echo "base image changed from ${OLD_BASE_ID} to ${NEW_BASE_ID}, rerunning base build"
    echo docker build -t cernml4reco/djcbase:$BASE_IMAGE_TAG -f Dockerfile_base .
    echo docker push --max-concurrent-uploads 3 cernml4reco/djcbase:$BASE_IMAGE_TAG
  fi
  
  echo docker build -t cernml4reco/deepjetcore3:latest . \
       --build-arg BUILD_DATE="$(date)" --build-arg BASE_IMAGE_TAG=$BASE_IMAGE_TAG
fi
