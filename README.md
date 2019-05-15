**For the bleeding-edge version of DeepJetCore check out the [updated installation](https://github.com/SwapneelM/DeepJetCore/blob/python-package/PYPKG.md) However, it is recommended to use the stable version described in the next Section**

*For details on the bleeding edge version, refer to the [presentations for DeepJet](https://drive.google.com/drive/folders/1l8Hu34hMYNc-YdgpCoAuqMzQ-qa5eCSJ?usp=sharing)*
*Check out the [DeepJetCore wiki](https://github.com/DL4Jets/DeepJetCore/wiki) for an introduction to the package.*

DeepJetCore: Package for training and evaluation of deep neural networks for HEP
===============================================================================

This package provides the basic functions for out-of-memory training, resampling, and basic evaluation. 
The actual training data structures and DNN models must be defined in an additional user package. The data structures (defining the structure of the training data as numpy arrays), must inherit from the TrainData class, and must be reachable in the PYTHONPATH as "from datastructure import * " .
A script to set it up will be provided eventually. For reference, please see: 
https://github.com/DL4Jets/DeepJet/tree/master/modules


Setup python packages (CERN)
==========
It is essential to perform all these steps on lxplus7. Simple ssh to 'lxplus7' instead of 'lxplus'

Pre-Installation: Anaconda setup (only once)
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
git clone https://github.com/DL4Jets/DeepJetCore
cd DeepJetCore/environment
./setup_djcenv.sh #opt: gpu
```
For enabling gpu support add 'gpu' (without quotes) as an additional option to the last command.
This will take a while. Please log out and in again once the installation is finised.

Compiling DeepJetCore
===========

When the installation was successful, the DeepJetCore tools need to be compiled.
```
cd <your working dir>
cd DeepJetCore
source env.sh
cd compiled
make -j4
```

After successfully compiling the tools, log out and in again.
The environment is set up.


Usage
==========

DeepJetCore is only a set of tools and wrappers and does not provide ready-to-use training code.
However, an example package containing more specific code examples (referred to as 'subpackage' in the following) can be created once everything is compiled and the environment is sourced (important!) using the script ``createSubpackage.py``.
This subpackage will include an example dataset which gets generated on the fly (size about 150 MB).
More instructions are printed by the script creating the subpackage.
This subpackage can serve as a reference for own projects.
In general, the following steps are needed for a training and evaluation:

  * Always source the environment script (```env.sh```) in the subpackage directory, NOT in DeepJetCore.
  * Define the training data structure, e.i. which branches from the input root ntuples are read, and how they are re-organised as input to the DNN. An example can be found in subpackage/modules/datastructures
  * Convert the root ntuples into the training data format using convertFromRoot.py. Please consider the help message for options (```convertFromRoot.py -h```). The input is a list of root files contained in a text file. An example dataset is generated when setting up the subpackage. It can be found in subpackage/example_data. Different files should be used for training and testing. To convert the training data, execute ```convertFromRoot.py -i <input text file> -o <output dir for training files> -c TrainData_example```, with TrainData_example being the data structure defined before. To create the test dataset, execute ```convertFromRoot.py -i <input text file> --testdatafor <output dir for training files>/dataCollection.dc -o <output dir for test data>```.
  * Train the model. The convertFromRoot.py script creates a set of output files and a descriptor (dataCollection.dc). This descriptor is fed to the training file which contains the model definition and an instance of the trainingbase class. An example is given in subpackage/Train/training_example.py. This file is called with ```python training_example.py /path/to/data/dataCollection.dc <output dir for the model etc.>```. More options are provided and described when calling ```python training_example.py -h```.
  * Once the training is done, the model can be used to predict the output of the model for the test data: ```predict.py /path/to/the/model.h5 /path/to/the/test/data/dataCollection.dc <output path for prediction>```. Please keep in mind that the output can become large (not the case in the example). 
  * For plotting, there are a few simple wrappers provided, which can be found in DeepJetCore/evaluation/evaluation.py, for making ROCs and simple plots. As input, these functions take the text file created by the predict.py script.


