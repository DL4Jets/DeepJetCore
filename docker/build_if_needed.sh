#!/bin/bash

FORCE="no"
if [ $1 ]
then 
FORCE=$1
fi

git fetch
if [ $(git rev-parse HEAD) != $(git rev-parse @{u}) ] || [ $FORCE != "no" ]
then


  COMMIT=$(git log -n 1 --pretty=format:"%H")
  
  echo "building container  for commit ${COMMIT}"
  
  OLD_BASE_ID=$(git rev-parse HEAD:docker/Dockerfile_base) 
  OLD_ID=$(git rev-parse HEAD:docker/Dockerfile) 
  git pull
  NEW_BASE_ID=$(git rev-parse HEAD:docker/Dockerfile_base) 
  NEW_ID=$(git rev-parse HEAD:docker/Dockerfile) 
  
  source image_tags.sh #in case this was updated in the pull
  
  BASE_IMAGE_TAG="${BASE_IMAGE_TAG}" # as this is a bleeding edge build
  
  if [ $OLD_BASE_ID != $NEW_BASE_ID ] || [ $FORCE == "force_base" ]
  then
    echo "base image changed from ${OLD_BASE_ID} to ${NEW_BASE_ID}, rerunning base build"
    docker build --no-cache=true -t cernml4reco/djcbase:$BASE_IMAGE_TAG -f Dockerfile_base .  > base_build.log 2>&1
    
    if [ $? != 0 ]; 
    then 
       BASE_FAIL=true
    else
       docker push cernml4reco/djcbase:$BASE_IMAGE_TAG  > base_push.log  2>&1
       if [ $? != 0 ]; 
       then
           BASE_PUSH_FAIL=true
       fi
    fi
    
    subject="Subject: base build ${BASE_IMAGE_TAG} finished"
    if [ $BASE_FAIL ]
    then
       subject="Subject: !! base build FAILED"
    fi
    if [ $BASE_PUSH_FAIL ]
    then
       subject="Subject: !! base push FAILED"
    fi
    
    { echo $subject ; 
      cat base_build.log ; 
      echo "" ;
      echo "################# push log ##############" ; 
      echo "" ;
      cat base_push.log ; } | sendmail jkiesele@cern.ch;
      
     FORCE_NO_CACHE=" --no-cache=true "
    
  fi
  
  if [ $BASE_FAIL ] || [ $BASE_PUSH_FAIL ]
  then
     exit
  fi
  
  # this is an auto build, so by definition, the build is not a release build
  # if the docker file changed, tag it as experimental and ask 
  TAG=latest
  if [ $OLD_ID != $NEW_ID ]
  then
      TAG=exp
  fi
  
  echo "Building with tag ${TAG}" > build.log
  
  # only force no cache if base image has been rebuilt
  docker build $FORCE_NO_CACHE -t cernml4reco/deepjetcore3:$TAG . \
       --build-arg BUILD_DATE="$(date)" --build-arg BASE_IMAGE_TAG=$BASE_IMAGE_TAG \
       --build-arg COMMIT=$COMMIT   >> build.log 2>&1
  if [ $? != 0 ]; 
  then 
     FAIL=true
  else
     
     docker push cernml4reco/deepjetcore3:$TAG  > push.log 2>&1
     
     if [ $? != 0 ]; 
     then
         PUSH_FAIL=true
     fi
  fi
    
  subject="Subject: build ${TAG} finished"
  if [ $FAIL ]
  then
     subject="Subject: !! DJC build FAILED"
  fi
  if [ $PUSH_FAIL ]
  then
     subject="Subject: !! DJC push FAILED"
     if [ $OLD_ID != $NEW_ID ]
     then
         subject="Subject: DJC experimental push failed (${TAG})"
     fi
  fi
  
  { echo $subject ; 
    cat build.log ; 
    echo "" ;
    echo "################# push log ##############" ; 
    echo "" ;
    cat push.log ; } | sendmail jkiesele@cern.ch;
      
      
fi
