#!/bin/bash

# hack to make root-numpy and ROOT work properly in python 
# this is necessary because the original package that is checked
# out by conda is broken

cd ${CONDA_PREFIX}
source bin/thisroot.sh
cd -

echo "Activate: ROOT has been sourced. Environment settings are ready. "
echo "ROOTSYS="${ROOTSYS}

if [ -n "${LD_LIBRARY_PATH}" ]; then
     unset LD_LIBRARY_PATH
fi


if [ -n "${DYLD_LIBRARY_PATH}" ]; then
     unset DYLD_LIBRARY_PATH
fi


