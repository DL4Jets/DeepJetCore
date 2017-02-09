#!/bin/bash
if [ ! `which conda` ]
then
echo Please install the anaconda package manager
exit 1
fi

conda env create -f conda_deepjet.yml
source activate deepjet
sed -i -e 's/CONDA_ENV_PATH/CONDA_PREFIX/g' $CONDA_PREFIX/etc/conda/activate.d/activateROOT.sh


echo "environment set up. Please activate it with \"source activate deepjet\""
