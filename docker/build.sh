#!/usr/bin/bash

# check if commit is given as argument, if not use the latest commit
if [ -z "$1" ]; then
    COMMIT=$(git rev-parse --short HEAD)
    # if we use the latest commit, assume we want to run on something we just did on a local repo.
    # check if we have uncommitted changes and if so warn the user.
    if [ -n "$(git status --porcelain)" ]; then
           echo "WARNING: You are building the latest commit, but you have uncommitted changes."
           echo "         This means that the image will not be reproducible."
           echo "         If you want to build the latest commit, please commit your changes first."
           read -p "Do you want to continue anyway? [y/N] " -n 1 -r
           echo
           if [[ ! $REPLY =~ ^[Yy]$ ]]; then
           exit 1
           fi
    fi
else
    COMMIT=$1
fi

# check if the commit has been pushed to https://github.com/DL4Jets/DeepJetCore
REMOTE_COMMIT=$(git ls-remote https://github.com/DL4Jets/DeepJetCore.git | grep $COMMIT)
if [ -z "$REMOTE_COMMIT" ]; then
       echo "ERROR: Commit $COMMIT not found in remote repository."
       exit 1
fi

# check if no-cache is given as argument
if [ "$2" == "--cache" ]; then
    FORCE_NO_CACHE=""
else
    FORCE_NO_CACHE="" #FIX
fi

#switch to directory this script is located in
cd "$(dirname "$0")"
echo "Building deepjetcore4:$COMMIT"
#print the current working directory
echo "Current working directory: $(pwd)"

docker build $FORCE_NO_CACHE -t cernml4reco/deepjetcore4:$COMMIT --build-arg BUILD_DATE="$(date)" --build-arg COMMIT=$COMMIT .
