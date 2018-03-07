#!/bin/bash
if [ ! `which conda` ]
then
echo Please install the anaconda package manager
exit 1
fi

if [ ! $1 ]
then
	echo "please specify an environment file"
	exit
fi

addstring=""

if [[ $2 == "gpu" ]]
then
	echo "setting up for gpu usage"
	addstring="_${2}"
fi
		
 

envfile=$1
envname="${envfile%.*}${addstring}"
pipfile="${envfile%.*}.pip"

conda create --copy --name $envname python=2.7.13 
conda install --name $envname --file $envfile


source activate $envname
pip install -r $pipfile

#conda install scikit-learn
#conda install numpy #to update packages. fast bugfix. make a new conda list later

cp activateROOT.sh  $CONDA_PREFIX/etc/conda/activate.d/activateROOT.sh 

if [ $addstring ]
then
	pip install --ignore-installed  --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.1-cp27-none-linux_x86_64.whl
        pip install setGPU
fi

echo "environment set up. Please activate it with \"source activate ${envname}\""

