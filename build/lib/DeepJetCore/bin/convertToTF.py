#!/usr/bin/env python

import imp
try:
    imp.find_module('setGPU')
    import setGPU
except ImportError:
    found = False
            
            
from keras.models import load_model
from argparse import ArgumentParser
from keras import backend as K
from Losses import * #needed!
import os


parser = ArgumentParser('')
parser.add_argument('inputModel')
parser.add_argument('outputDir')
args = parser.parse_args()

 
if os.path.isdir(args.outputDir):
    raise Exception('output directory must not exists yet')


model=load_model(args.inputModel, custom_objects=global_loss_list)
import tensorflow as tf
import keras.backend as K
tfsession=K.get_session()
saver = tf.train.Saver()
tfoutpath=args.outputDir+'/tf'
import os
os.system('mkdir -p '+tfoutpath)
saver.save(tfsession, tfoutpath)


