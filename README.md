

DeepJet: Repository for training and evaluation of deep neural networks for HEP
===============================================================================


Setup (CERN)
==========
It is essential to perform all these steps on lxplus7. Simple ssh to 'lxplus7' instead of 'lxplus'

Pre-Installtion: Anaconda setup (only once)
Download miniconda3
```
cd <afs work directory: you need some disk space for this!>
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
Please follow the installation process. If you don't know what an option does, please answer 'yes'.
After installation, you have to log out and log in again for changes to take effect.
If you don't use bash, you might have to add the conda path to your .rc file
```
export PATH="<your miniconda directory>/miniconda3/bin:$PATH"
```
This has to be only done once.


Installation:

```
mkdir <your working dir>
cd <your working dir>
git clone https://github.com/mstoye/DeepJet
cd DeepJet/environment
./setupEnv.sh deepjetLinux3.conda
```
For enabling gpu support add 'gpu' as an additional option to the last command.
This will take a while. Please log out and in again once the installation is finised.

When the installation was successful, the DeepJet tools need to be compiled.
```
cd <your working dir>
cd DeepJet/environment
source lxplus_env.sh
cd ../modules
make -j4
```

After successfully compiling the tools, log out and in again.
The environment is set up.


Usage
==============

After logging in, please source the right environment (please cd to the directory first!):
```
cd <your working dir>/DeepJet/environment
source setup_env.sh
```


The preparation for the training consists of the following steps
====

- define the data structure for the training (example in modules/TrainData_topreco.py)
  for simplicity, copy the file to TrainData_mytopreco.py and adjust it. 
  Define a new class name (e.g. TrainData_mytopreco), leave the inheritance untouched
  
- register this class in DeepJet/convertFromRoot/convertFromRoot.py by 
  a) importing it (the line in the code is indiacted by a comment)
  b) adding it to the class list below

- convert the root file to the data strucure for training:
  ```
  cd DeepJet/convertFromRoot
  ./convertFromRoot.py -i /path/to/the/root/ntuple/list_of_root_files.txt -o /output/path/that/needs/some/disk/space -c TrainData_mytopreco
  ```
  
  This step can take a while.


- prepare the training file and the model. Please refer to DeepJet/Train/XXX_template.reference.py
  


Training
====

Since the training can take a while, it is advised to open a screen session, such that it does not die at logout.
```
ssh lxplus.cern.ch
<note the machine you are on, e.g. lxplus058>
screen
ssh lxplus7
```
Then source the environment, and proceed with the training. Detach the screen session with ctr+a d.
You can go back to the session by logging in to the machine the session is running on (e.g. lxplus58):

```
ssh lxplus.cern.ch
ssh lxplus058
screen -r
``` 

Please close the session when the training is finished

the training is launched in the following way:
```
python train_template.py /path/to/the/output/of/convert/dataCollection.dc <output dir of your choice>
```


Evaluation
====

After the training has finished, the performance can be evaluated.
The evaluation consists of a few steps:

1) converting the test data
```
cd DeepJet/convertFromRoot
./convertFromRoot.py --testdatafor <output dir of training>/trainsamples.dc -i /path/to/the/root/ntuple/list_of_test_root_files.txt -o /output/path/for/test/data
```

2) applying the trained model to the test data
```
predict.py <output dir of training>/KERAS_model.py  /output/path/for/test/data/dataCollection.dc <output directory>
```
This creates output trees. and a tree_association.txt file that is input to the plotting tools

There is a set of plotting tools with examples in 
DeepJet/Train/Plotting


