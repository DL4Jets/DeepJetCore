#!/usr/bin/env python

import imp
from DeepJetCore.DataCollection import DataCollection
try:
    imp.find_module('setGPU')
    import setGPU
except ImportError:
    found = False
    
from keras.models import load_model
from DeepJetCore.DJCLosses import *
from DeepJetCore.DJCLayers import *
from argparse import ArgumentParser
from keras import backend as K
import os
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


parser = ArgumentParser('Apply a model to a (test) source sample and create friend trees to inject it inthe original ntuple')
parser.add_argument('inputModel')
parser.add_argument('trainingDataCollection')
parser.add_argument('inputSourceFileList')
parser.add_argument('outputDir')

args = parser.parse_args()

 
#if os.path.isdir(args.outputDir):
#    raise Exception('output directory must not exists yet')

custom_objs = {}
custom_objs.update(djc_global_loss_list)
custom_objs.update(djc_global_layers_list)
custom_objs.update(global_loss_list)
custom_objs.update(global_layers_list)
custom_objs.update(global_metrics_list)


model=load_model(args.inputModel, custom_objects=custom_objs)
dc = DataCollection(args.trainingDataCollection)
td = dc.dataclass
outputs = []
inputdir = os.path.abspath(os.path.dirname(args.inputSourceFileList))
os.system('mkdir -p '+args.outputDir)

with open(args.inputSourceFileList, "r") as f:
    for inputfile in f:
        inputfile = inputfile.replace('\n', '')
        print('converting '+inputfile)
        x, y, w = td.convertFromSourceFile(inputdir+"/"+inputfile, dc.weighterobjects, istraining=False)
        outfilename = "pred_"+inputfile
        print('predicting '+inputfile)
        predicted = model.predict(x)
        if not type(predicted) == list: #circumvent that keras return only an array if there is just one list item
            predicted = [predicted]   
        td.writeOutPrediction(predicted, x, y, w, args.outputDir + "/" + outfilename, inputfile)
        outputs.append(outfilename)
        
    
    
with open(args.outputDir + "/outfiles.txt","w") as f:
    for l in outputs:
        f.write(l+'\n')
    
    
    
    
    
