#!/usr/bin/env python


#script that takes model in .h5 format as input as spits out the graph format used in CMSSW

import imp
try:
    imp.find_module('setGPU')
    import setGPU
except ImportError:
    found = False
            
import tensorflow as tf

from DeepJetCore.DJCLosses import *
from DeepJetCore.DJCLayers import *
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


custom_objs = {}
custom_objs.update(djc_global_loss_list)
custom_objs.update(djc_global_layers_list)
custom_objs.update(global_loss_list)
custom_objs.update(global_layers_list)
custom_objs.update(global_metrics_list)

sess = tf.Session()
from keras.models import load_model
from argparse import ArgumentParser
from keras import backend as K
from Losses import * #needed!
import os

K.set_session(sess)

parser = ArgumentParser('')
parser.add_argument('inputModel')
parser.add_argument('outputDir')
args = parser.parse_args()

 
if os.path.isdir(args.outputDir):
    raise Exception('output directory must not exists yet')

model=load_model(args.inputModel, custom_objects=custom_objs)

K.set_learning_phase(0)
inputs = [node.op.name for node in model.inputs]
print ("input layer names", inputs)
outputs = [node.op.name for node in model.outputs]
print ("output layer names",outputs)
constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), outputs)
tfoutpath=args.outputDir+'/tf'
import os
os.system('mkdir -p '+tfoutpath)
tf.train.write_graph(constant_graph, tfoutpath, "constant_graph.pb", as_text=False)


