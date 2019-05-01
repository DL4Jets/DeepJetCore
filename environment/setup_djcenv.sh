#!/bin/bash
if [ ! `which conda` ]
then
echo Please install the anaconda package manager
exit 1
fi


addstring=""

if [[ $1 == "gpu" ]]
then
	echo "setting up for gpu usage"
	addstring="_${1}"
fi
		
 

envfile=djcenv.conda
envname="${envfile%.*}${addstring}"
pipfile="${envfile%.*}.pip"
pipfilegpu="${envfile%.*}_gpu.pip"


conda create --copy --name $envname python=2.7.13 
conda install --name $envname --file $envfile

source activate $envname

if [ $addstring ]
then
   pip install -r $pipfilegpu
   pip install setGPU
else
   pip install -r $pipfile
fi


cp activateROOT.sh  $CONDA_PREFIX/etc/conda/activate.d/activateROOT.sh 


echo "environment set up. Please activate it with \"source activate ${envname}\""

