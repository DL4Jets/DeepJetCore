#!/usr/bin/env python

import imp
try:
    imp.find_module('setGPU')
    import setGPU
except ImportError:
    found = False
    
from keras.models import load_model
from DeepJetCore.evaluation import testDescriptor
from argparse import ArgumentParser
from keras import backend as K
import imp
try:
    imp.find_module('Losses')
    from Losses import *
except ImportError:
    print 'No Losses module found, ignoring at your own risk'
    global_loss_list = {}

try:
    imp.find_module('Layers')
    from Layers import *
except ImportError:
    print 'No Layers module found, ignoring at your own risk'
    global_layers_list = {}

try:
    imp.find_module('Metrics')
    from Metrics import *
except ImportError:
    print 'No metrics module found, ignoring at your own risk'
    global_metrics_list = {}

import os


parser = ArgumentParser('Apply a model to a (test) sample and create friend trees to inject it inthe original ntuple')
parser.add_argument('inputModel')
parser.add_argument('inputDataCollection')
parser.add_argument('outputDir')
parser.add_argument('--use', help='coma-separated list of prediction indexes to be used')
parser.add_argument('--labels', action='store_true', help='store true labels in the trees')
parser.add_argument('--monkey_class', default='', help='allows to read the data with a different TrainData, it is actually quite dangerous if you do not know what you are doing')
parser.add_argument('--numpy', help='switches on numpy rec-array output in addition to root files', action='store_true' , default=False )

args = parser.parse_args()

 
if os.path.isdir(args.outputDir):
    raise Exception('output directory must not exists yet')

custom_objs = {}
custom_objs.update(global_loss_list)
custom_objs.update(global_layers_list)
custom_objs.update(global_metrics_list)
model=load_model(args.inputModel, custom_objects=custom_objs)


td=testDescriptor(addnumpyoutput = args.numpy)
if args.use:
	td.use_only = [int(i) for i in args.use.split(',')]

from DeepJetCore.DataCollection import DataCollection

testd=DataCollection()
testd.readFromFile(args.inputDataCollection)


os.mkdir(args.outputDir)

td.makePrediction(
    model, testd, args.outputDir,
    store_labels = args.labels,
    monkey_class = args.monkey_class
)

td.writeToTextFile(args.outputDir+'/tree_association.txt')

#    make the file reading entirely C++
#    then it can be used for other studies
