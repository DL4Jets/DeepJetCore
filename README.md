Notes
============
Due to a recent update of uproot, please change ``import uproot`` to ``import uproot3`` in case you have issues with the import and the newest containers.



DeepJetCore: Package for training and evaluation of deep neural networks for HEP
===============================================================================


(For upgrading from DeepJetCore 1.X to 2.X/3.X, please scroll to the bottom)

This package provides the basic functions for out-of-memory training, resampling, and basic evaluation. 
For simple use cases, only two files need to be adapted: the actual training data structures, describing how to fill numpy arrays from root trees, and the DNN model itself. Both must be defined in an additional user package. 
Please refer to the Section 'Usage' for more information.

**If you use DeepJetCore or the containers, please mention it in (internal) documentation and talks and cite the zenodo entry:**

https://zenodo.org/record/3670882/export/hx


Setup
==========

The package comes with a docker file in the subdirectory docker, which will set up a container with the needed environment.


**Users with access to cvmfs** (e.g. on lxplus or other CERN machines) will find a pre-built singularity container here:
``/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cernml4reco/deepjetcore3:latest``

A good way to enter the container interactively, mounting for example the standard directories on lxplus or similar machines would be:
```
#!/bin/bash

gpuopt=""
files=$(ls -l /dev/nvidia* 2> /dev/null | egrep -c '\n')
if [[ "$files" != "0" ]]
then
gpuopt="--nv"
fi

#this is a singularity problem only fixed recently
unset LD_LIBRARY_PATH
unset PYTHONPATH
sing=`which singularity`
unset PATH
cd

$sing run -B /eos -B /afs $gpuopt /cvmfs/unpacked.cern.ch/registry.hub.docker.com/cernml4reco/deepjetcore3:latest
```

The cache dir can get rather large and is normally located at ~/.singularity/cache. To avoid filling up the home afs, the cache can be set to /tmp or the work afs. Once the container is fully closed, the cache can be safely deleted. Singularity reacts to environment variables, e.g.

```
export SINGULARITY_CACHEDIR="/tmp/$(whoami)/singularity"
```

Sometimes you need to try a few times - singularity is a bit weird. But once the container is launched, everything works smoothly.
The message about a missing user group can be safely ignored.

**It is important** that your bashrc does not change or reset the ``LD_LIBRARY`` or ``PYTHONPATH`` environment variables. Also **remove any anaconda paths from your bashrc**, because they will reset ``LD_LIBRARY`` and ``PYTHONPATH``. The system needs to be in a clean environment state within the container (as it should be).


**Building / using the container** 
For users without access to cvmfs, the container can be built manually or pulled from dockerhub. For every release, there is a container on dockerhub, the latest release is tagged. The ``latest`` tag refers to the developing master branch.
``cernml4reco/deepjetcore3:latest``
It can be pulled using docker, or pulled from singularity. For details, please see the corresponding docker or singularity documentation.
To build the container, the files can be found in the ``docker`` subdirectory. It is a two-stage process. First the container ``Dockerfile_base`` needs to be built, containing the basic system packages. This will take a while, because root is being compiled within. The output container name must be ``cernml4reco/djcbase:cu11.1``. In the next step, the actual DeepJetCore container should be built on top. **Please make sure to use a release, not the developing master branch!**


**Building with conda**
In case of lack of access to singularity or docker, it's possible to build the environment using conda. This _should_ work anywhere since conda
pulls tensorflow's CUDA dependencies.;
```
git clone https://github.com/DL4Jets/DeepJetCore.git
cd DeepJetCore
# This will take some time (includes ROOT, tensorflow-gpu, CUDA...)
conda env create -f conda_env.yml  # if that fails use conda_env_exact.yml with entirely fixed versions
conda activate DJC2
# Build compiled modules with a modified Makefile
source docker_env.sh # sets up paths, never mind the name
cd compiled 
make -f Makefile_conda -j4 # conda needs slightly different lib names
```


Usage
==========

DeepJetCore is only a set of tools and wrappers and does not provide ready-to-use training code.
However, an example package containing more specific code examples (referred to as 'subpackage' in the following) can be created once the container is launched using the script ``createSubpackage.py``.
This subpackage includes an example dataset which gets generated on the fly (size about 150 MB) using the ``--data`` option.
**More instructions are printed by the script creating the subpackage and serve as documentation for a simple training.**
This subpackage can serve as a reference for own projects.
In general, the following steps are needed for a training and evaluation:

  * Always source the environment script (``env.sh``) in the subpackage directory, NOT in DeepJetCore.
  * Define the training data structure, e.i. which branches from the input root ntuples are read, and how they are re-organised as input to the DNN. An example can be found in subpackage/modules/datastructures
  * Convert the root ntuples into the training data format using convertFromSource.py. Please consider the help message for options (``convertFromSource.py -h``). The input is a list of root files contained in a text file. An example dataset is generated when setting up the subpackage. It can be found in subpackage/example_data. 
  Different files should be used for training and testing. To convert the training data, execute ``convertFromSource.py -i <input text file> -o <output dir for training files> -c TrainData_example``, with TrainData_example being the data structure defined before. The test data will be directly read by from the source files (see below).
  * Train the model. The convertFromSource.py script creates a set of output files and a descriptor (dataCollection.djcdc). This descriptor is fed to the training file which contains the model definition and an instance of the trainingbase class. An example is given in subpackage/Train/training_example.py. 
  This file is called with ``python3 training_example.py /path/to/data/dataCollection.djcdc <output dir for the model etc.>``. More options are provided and described when calling ``python3 training_example.py -h``.
  * Once the training is done, the model can be used to predict the output of the model for the test data: ``predict.py /path/to/the/model.h5 /path/to/the/training/dataCollection.djcdc <text file containing list of source test files> <output directoy for prediction>``. Please keep in mind that the output can become large (not the case in the example). 
  * For plotting, there are a few simple wrappers provided, which can be found in DeepJetCore/evaluation/evaluation.py, for making ROCs and simple plots. As input, these functions take the text file created by the predict.py script.


The general pipeline for training is depicted in the following sketch:
![pipeline](https://github.com/DL4Jets/DeepJetCore/blob/master/training_pipeline.png "Data pipeline for training")


The general pipeline for inference/prediction is depicted in the following sketch:
![pipeline](https://github.com/DL4Jets/DeepJetCore/blob/master/predict_pipeline.png "Data pipeline for prediction")

More information on the three function of TrainData that need to be defined by the user (in addition to the training script) is given in the next Section.
For the training script, please refer to the example provided with ``createSubpackage.py``. 

TrainData definition and notes on upgrading (from 1.X to 2.X/3.X)
=========================

There has been substantial format changes from 1.X to 2.X, including low-level support preparations for ragged tensors. Therefore, all data from 1.X needs to be converted or newly created. Also, the interface changed slightly.

The master branch has also been switched to tensorflow 2.0 recently. This might require adapting subpackages. A workaround for the moment for old subpackages would be to include the following code in the subpackage ``__init__.py``:

```
import sys
import tensorflow 
tensorflow.compat.v1.disable_eager_execution()
sys.modules["tensorflow"]=tensorflow.compat.v1
```

The TrainData class has been slimmed significantly. Now, the ``__init__`` function does not need any additional arguments anymore, and there are no mandatory definitions. Only the following functions should be defined for the interface (all others are deprecated):

  * ``createWeighterObjects(self, allsourcefiles)``: is not mandatory. It can be used, however to create a dictionary (pickable) objects that depend on the whole dataset (e.g. for numbers for normalisation etc). **Returns**: a dictionary of weighter objects
  * ``convertFromSourceFile(self, filename, weighterobjects, istraining, **kwargs)``: is mandatory. This function defines a rule to convert one source file to one output file. The final output should be a list of numpy feature arrays, a list of numpy truth arrays, and a list of numpy weight arrays. The latter can also be empty. The conversion can be done from root e.g. with uproot or similar, but can also use any other input format. **Returns** three items: a list of feature arrays, list of truth arrays, list of weight arrays
  * ``writeOutPrediction(self, predicted, features, truth, weights, outfilename, inputfile)``: is mandatory. Defines how the output of the network should be written back to an output format. This could e.g. be a root tree, which can be a friend to the original tree, or any other output. The function gives optional access to all input features, truth, weights (if any), and the input source file name. **Returns**: nothing

Of course any user function, member etc beyond that can be defined, too.

For everybody who used the built-in weighting functionality, this function now need to be reimplemented in the user code (in ``createWeighterObjects``). The old implementation can be found here: https://github.com/DL4Jets/DeepJetCore/blob/1.X/TrainData.py#L657-L672 and can be used in the same way in the user code.

Ragged Tensors
=====================

Ragged data structures supported to some extent with workarounds for places where either keras or tensorflow are still missing support.
Only tensors ragged in the first dimension are supported so far, meaning, e.g. a different number of inputs per event, but each input having the same feature length (or more dimensions with fixed sizes). This should cover most usecases already.
To create a ragged data structure, the function ``convertFromSourceFile(self, filename, weighterobjects, istraining, **kwargs)`` must not return a list of numpy arrays, but a list of ``DeepJetCore.SimpleArray``, which is a DeepJetCore class supporting ragged structures. This array can be constructed from two numpy arrays: one describing the row splits (as int, in the same format as tensorflow ragged array row splits) and one containing the data, where the 0th and 1st dimensions are flattened. Then the array can be filled by invoking ``createFromNumpy(data, row_splits``.
For training, the model will receive a list of tensors, where the first one will be the data and the second the row splits. The same applies to the truth. This circumvents incomplete keras support for the moment.


Using the data pipeline directly
=====================

The data pipeline can also be used without the training_base class sourrounding it. The DataCollection can be opened and a generator function can be invoked that returns the data batch-by-batch. An example is given here:

```
from DeepJetCore.DataCollection import DataCollection
train_data = DataCollection("path/to/your/dataCollection.dc")

# splits off 10% of the training dataset for validation. Can be used in the same way as train_data
val_data=train_data.split(0.9) 

# Set the batch size. 
# If the data is ragged in dimension 1 (see convert options), 
# then this is the maximum number of elements per batch, which could be distributed differently
# to individual examples. E.g., if the first example has 50 elements, the second 48, and the third 30,
# and the batch size is set to 100, it would return the first two examples (in total 99 elements) in 
# the first batch etc. This is helpful to avoid out-of-memory errors during training

train_data.setBatchSize(100) 

# prepare the generator

gen = train_data.invokeGenerator()

# loop over epochs here ...

gen.shuffleFileList()
gen.prepareNextEpoch()

# this number can differ from epoch to epoch for ragged data!
nbatches = gen.getNBatches()
generator=gen.feedNumpyData()

for b in range(nbatches):
    
    #should not happen unless files are broken (will give additional errors)
    if gen.isEmpty():
        raise Exception("ran out of data") 
    
    # weights are optional, each of these is a list of numpy arrays
    features_list, truth_list,  weight_list = next(generator)
    
    # do your training
    
    
# end epoch loop

```


For developers
=====================

For developing the best strategy is to create a container or just run it, and then redirect the paths etc to the development directory. This can be done conveniently by sourcing the file ``docker_env.sh``. After sourcing the environment, the package needs to be compiled by running ``make -j5`` in the ``compiled`` directory.




