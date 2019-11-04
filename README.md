

DeepJetCore: Package for training and evaluation of deep neural networks for HEP
===============================================================================

This package provides the basic functions for out-of-memory training, resampling, and basic evaluation. 
For simple use cases, only two files need to be adapted: the actual training data structures, describing how to fill numpy arrays from root trees, and the DNN model itself. Both must be defined in an additional user package. 
Please refer to the Section 'Usage' for more information.


Setup
==========

The package comes with a docker file in the subdirectory docker, which will set up a container with the needed environment.


**Users with access to Cernbox** can just run the prepared script at:
``/eos/home-j/jkiesele/singularity/run_deepjetcore.sh``
Please run this script from your home directory. Sometimes you need to try two or three times - singularity is a bit weird. But once the contaienr is launched, everything works smoothly.
The message about a missing user group can be safely ignored.


Usage
==========

DeepJetCore is only a set of tools and wrappers and does not provide ready-to-use training code.
However, an example package containing more specific code examples (referred to as 'subpackage' in the following) can be created once everything is compiled and the environment is sourced (important!) using the script ``createSubpackage.py``.
This subpackage will include an example dataset which gets generated on the fly (size about 150 MB).
More instructions are printed by the script creating the subpackage.
This subpackage can serve as a reference for own projects.
In general, the following steps are needed for a training and evaluation:

  * Always source the environment script (``env.sh``) in the subpackage directory, NOT in DeepJetCore.
  * Define the training data structure, e.i. which branches from the input root ntuples are read, and how they are re-organised as input to the DNN. An example can be found in subpackage/modules/datastructures
  * Convert the root ntuples into the training data format using convertFromRoot.py. Please consider the help message for options (``convertFromRoot.py -h``). The input is a list of root files contained in a text file. An example dataset is generated when setting up the subpackage. It can be found in subpackage/example_data. 
  Different files should be used for training and testing. To convert the training data, execute ``convertFromRoot.py -i <input text file> -o <output dir for training files> -c TrainData_example``, with TrainData_example being the data structure defined before. To create the test dataset, execute ``convertFromRoot.py -i <input text file> --testdatafor <output dir for training files>/dataCollection.dc -o <output dir for test data>``.
  * Train the model. The convertFromRoot.py script creates a set of output files and a descriptor (dataCollection.dc). This descriptor is fed to the training file which contains the model definition and an instance of the trainingbase class. An example is given in subpackage/Train/training_example.py. 
  This file is called with ``python training_example.py /path/to/data/dataCollection.dc <output dir for the model etc.>``. More options are provided and described when calling ``python training_example.py -h``.
  * Once the training is done, the model can be used to predict the output of the model for the test data: ``predict.py /path/to/the/model.h5 /path/to/the/test/data/dataCollection.dc <output path for prediction>``. Please keep in mind that the output can become large (not the case in the example). 
  * For plotting, there are a few simple wrappers provided, which can be found in DeepJetCore/evaluation/evaluation.py, for making ROCs and simple plots. As input, these functions take the text file created by the predict.py script.


