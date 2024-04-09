#!/usr/bin/bash

# check if commit is given as argument, if not use the latest commit
if [ -z "$1" ]; then
    COMMIT=$(git rev-parse --short HEAD)
else
    COMMIT=$1
fi

# check if no-cache is given as argument
if [ "$2" == "--cache" ]; then
    FORCE_NO_CACHE=""
else
    FORCE_NO_CACHE="--no-cache"
fi

#switch to directory this script is located in
cd "$(dirname "$0")"
echo "Building deepjetcore4:$COMMIT"
#print the current working directory
echo "Current working directory: $(pwd)"

docker build $FORCE_NO_CACHE -t cernml4reco/deepjetcore4:$COMMIT --build-arg BUILD_DATE="$(date)" --build-arg COMMIT=$COMMIT .