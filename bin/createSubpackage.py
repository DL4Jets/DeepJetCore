#!/usr/bin/env python
# encoding: utf-8


import sys
import os
from argparse import ArgumentParser

parser = ArgumentParser('script to create a DeepJetCore subpackage')

parser.add_argument("subpackage_name", help="name of the subpackage")
parser.add_argument("subpackage_parent_dir", help="parent directory of the subpackage (must be same as DeepJetCore)")
args=parser.parse_args()

deepjetcore = os.getenv('DEEPJETCORE')

subpackage_dir=args.subpackage_parent_dir+'/'+args.subpackage_name

### templates ####

environment_file='''
#! /bin/bash
THISDIR=`pwd`
export {subpackage}=$( cd "$( dirname "${BASH_SOURCE}" )" && pwd -P)
export DEEPJETCORE_SUBPACKAGE=${subpackage}
cd {deepjetcore}
if command -v nvidia-smi > /dev/null
then
        source gpu_env.sh
else
        source lxplus_env.sh
fi
cd ${subpackage}
export PYTHONPATH=${subpackage}/modules:$PYTHONPATH
export PYTHONPATH=${subpackage}/modules/datastructures:$PYTHONPATH
'''.format(deepjetcore=deepjetcore, 
           subpackage=args.subpackage_name.upper(),
           subpackage_dir=os.path.abspath(subpackage_dir),
           BASH_SOURCE="{BASH_SOURCE[0]}")

create_dir_structure_script='''
#! /bin/bash
mkdir -p {subpackage_dir}
mkdir -p {subpackage_dir}/modules
mkdir -p {subpackage_dir}/modules/datastructures
mkdir -p {subpackage_dir}/scripts
mkdir -p {subpackage_dir}/Train
mkdir -p {subpackage_dir}/example_data
'''.format(subpackage_dir=subpackage_dir)

datastructure_template='''
from DeepJetCore.TrainData import TrainData
import numpy 

class TrainData_template(TrainData):
    def __init__(self):
        TrainData.__init__(self)

        self.treename="tree" #input root tree name
        
        self.truthclasses=[''] #truth classes for classification
        
        self.weightbranchX='branchx' #needs to be specified
        self.weightbranchY='branchy' #needs to be specified
        
        self.referenceclass='flatten'
        self.weight_binX = numpy.array([0,-40000,40000],dtype=float) 
        self.weight_binY = numpy.array([-40000,40000],dtype=float) 
        
        
        self.registerBranches(['']) #list of branches to be used 
        
        self.registerBranches(self.truthclasses)
        
        
        #call this at the end
        self.reduceTruth(None)
        
    
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
    
        # this function defines how to convert the root ntuple to the training format
        # options are not yet described here
        
        
        # user code
        feature_array = function_to_create_the_array(filename)
        
        notremoves=weighter.createNotRemoveIndices(Tuple)
        
        # this removes parts of the dataset for weighting the events
        feature_array = feature_array[notremoves > 0]
                
        # call this in the end
        
        self.nsamples=len(feature_array)
        
        self.x=[] # list of feature numpy arrays
        self.y=[] # list of target numpy arrays (truth)
        self.w=[] # list of weight arrays. One for each truth target

'''


training_template='''
from DeepJetCore.training.training_base import training_base
import keras
from keras.models import Model
from keras.layers import Dense #etc

def my_model(Inputs,nclasses,nregressions,otheroption):
    
    input_a = Inputs[0] #this is the self.x list from the TrainData data structure
    x = Dense(2)(input_a)
    predictions = [x]
    return Model(inputs=Inputs, outputs=predictions)


train=training_base(testrun=False,resumeSilently=False,renewtokens=True)


if not train.modelSet(): # allows to resume a stopped/killed training. Only sets the model if it cannot be loaded from previous snapshot

    train.setModel(my_model,otheroption=1)
    
    train.compileModel(learningrate=0.01,
                   loss='mean_squared_error') 
                   

model,history = train.trainModel(nepochs=10, 
                                 batchsize=100,
                                 checkperiod=1, # saves a checkpoint model every N epochs
                                 verbose=1)

'''
        
layers_template='''
# Define custom layers here and add them to the global_layers_list dict (important!)
global_layers_list = {}
'''
losses_template='''
# Define custom losses here and add them to the global_loss_list dict (important!)
global_loss_list = {}
'''

metrics_template='''
# Define custom metrics here and add them to the global_metrics_list dict (important!)
global_metrics_list = {}
'''


######## create the structure ########


os.system(create_dir_structure_script)
with  open(subpackage_dir+'/env.sh','w') as envfile:
    envfile.write(environment_file)
    
with  open(subpackage_dir+'/modules/datastructures/TrainData_template.py','w') as lfile:
    lfile.write(datastructure_template)
    
with  open(subpackage_dir+'/Train/training_template.py','w') as lfile:
    lfile.write(training_template)
    
with  open(subpackage_dir+'/modules/Layers.py','w') as lfile:
    lfile.write(layers_template)
with  open(subpackage_dir+'/modules/Losses.py','w') as lfile:
    lfile.write(losses_template)
with  open(subpackage_dir+'/modules/Metrics.py','w') as lfile:
    lfile.write(metrics_template)


















