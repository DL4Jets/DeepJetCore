from __future__ import absolute_import
from __future__ import division
from pdb import set_trace
from argparse import ArgumentParser
# argument parsing and bookkeeping
parser = ArgumentParser('Run the training')
parser.add_argument('inputfile')
parser.add_argument('outputDir')
parser.add_argument('-f','--force', action='store_true', help='overwrite the directory if there')
parser.add_argument('--class_only', action='store_true', help='run only classification')
parser.add_argument('--warm', help='pre-trained model')
args = parser.parse_args()

import sys
print 'Command issued:'
print ' '.join(sys.argv)

import matplotlib
#if no X11 use below
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.layers import Input
#zero padding done before
#from keras.layers.convolutional import Cropping1D, ZeroPadding1D
from keras.optimizers import SGD

## to call it from cammand lines
import sys
import os
import shutil

import sys
print 'got command: %s' % ' '.join(sys.argv)

from glob import glob
outputDir = args.outputDir
# create output dir

if os.path.isdir(outputDir) and not args.force:
    print('output directory must not exists yet')
    raise Exception('output directory must not exists yet')

if not os.path.isdir(outputDir):
   os.mkdir(outputDir)

outputDir = os.path.abspath(outputDir)
outputDir+='/'

#copy configuration to output dir

shutil.copyfile(sys.argv[0],outputDir+sys.argv[0])
shutil.copyfile('../modules/DeepJet_models.py',outputDir+'DeepJet_models.py')

######################### KERAS PART ######################

# configure the in/out/split etc
config_args = { #we might want to move it to an external file
   'testrun'   : False,
   'nepochs'   : 100,
   'batchsize' : 2000,
   'startlearnrate' : 0.0005,
   'useweights' : False,
   'splittrainandtest' : 0.8,
   'maxqsize' : 50, #sufficient
   'conv_dropout' : 0.1,
   'loss_weights' : [1., .025] ,
}

from DeepJet_callbacks import DeepJet_callbacks
callbacks = DeepJet_callbacks(
   #early stopping patience                            
   stop_patience=300, 
   #learning rate reduction
   lr_factor=0.5,
   lr_patience=2, 
   lr_epsilon=0.003, 
   lr_cooldown=6, 
   lr_minimum=0.000001,                             
   #check point outputs
   outputDir=outputDir
)


from DataCollection import DataCollection
from TrainData_deepCSV_PF_binned import TrainData_deepCSV_PF_Binned

traind = DataCollection(args.inputfile)
traind.useweights = config_args['useweights']

if config_args['testrun']:
   traind.split(0.02)
   nepochs=10
    
testd = traind.split(config_args['splittrainandtest'])
inputs_shapes = traind.getInputShapes()
output_shapes = traind.getTruthShape()

inputs = (
    Input(shape=inputs_shapes[0]),
    Input(shape=inputs_shapes[1]),
    Input(shape=inputs_shapes[2]),
    Input(shape=inputs_shapes[3]),
    Input(shape=inputs_shapes[4])
)

####################################################
#                                                  #
#                    MODEL                         #
#                                                  #
####################################################

from DeepJet_models import binned3D_convolutional_classification_regression
model = binned3D_convolutional_classification_regression(
    inputs, output_shapes, config_args['conv_dropout'],
    run_regression=(not args.class_only)
)

####################################################
#                                                  #
#         Compiling model and setting stuff        #
#                                                  #
####################################################

print 'compiling'
from Losses import loss_NLL

from keras.optimizers import Adam
adam = Adam(lr = config_args['startlearnrate'])
if args.class_only:
    model.compile(
        loss = 'categorical_crossentropy',
        optimizer = adam, metrics = ['categorical_accuracy']
        )
else:
    model.compile(
        loss = ['categorical_crossentropy', loss_NLL], #apply xentropy to the first output (flavour) and NLL to the pt regression
        optimizer = adam, metrics = ['categorical_accuracy','accuracy'],
        loss_weights = config_args['loss_weights']
        )

ntrainepoch = traind.getSamplesPerEpoch()
nvalepoch   = testd.getSamplesPerEpoch()

testd.isTrain  = False
traind.isTrain = True

####################################################
#                                                  #
#                Warm start model                  #
#                                                  #
####################################################

if args.warm:
    print 'warm start from model directory %s' % args.warm
    from keras.models import load_model
    #model  = load_model('%s/KERAS_check_last_model.h5' % args.warm)
    model.load_weights('%s/KERAS_check_last_model.h5' % args.warm)
    shutil.copyfile('%s/trainsamples.dc' % args.warm, '%s/trainsamples.dc' % args.inputDir)
    shutil.copyfile('%s/valsamples.dc' % args.warm, '%s/valsamples.dc' % args.inputDir)
    traind = DataCollection('%s/trainsamples.dc' % args.inputDir)
    testd  = DataCollection('%s/valsamples.dc'   % args.inputDir)


print 'split to %d train samples and %d test samples' % (ntrainepoch, nvalepoch)
#for bookkeeping
traind.setBatchSize(config_args['batchsize'])
testd.setBatchSize(500)
traind.writeToFile(outputDir+'trainsamples.dc')
testd.writeToFile( outputDir+'valsamples.dc')

print 'training'


####################################################
#                                                  #
#                The real training                 #
#                                                  #
####################################################

def flav_only(generator):
    for X, Y in generator:
        yield X, Y[0]#.astype(int)


model.fit_generator(
   flav_only(traind.generator()) if args.class_only else traind.generator(), 
   verbose=1,
   steps_per_epoch = traind.getNBatchesPerEpoch(), 
   epochs = config_args['nepochs'],
   callbacks = callbacks.callbacks,
   validation_data = flav_only(testd.generator()) if args.class_only else testd.generator(),
   validation_steps = testd.getNBatchesPerEpoch(),
   max_q_size = config_args['maxqsize'], #maximum size for the generator queue
   class_weight = 'auto' if args.class_only else None,
   )

#Loss for classification should be < 0.48 regression loss < 2.8
#epochs for final training 100, for testing 10

####################################################
#                                                  #
#           Plots, still to be fixed               #
#                                                  #
####################################################

model.save(outputDir+"KERAS_model.h5")
traind.writeToFile(outputDir+'trainsamples.dc')
testd.writeToFile(outputDir+'valsamples.dc')

# summarize history for loss for trainin and test sample
plt.plot(callbacks.history.history['loss'])
#print(callbacks.history.history['val_loss'],history.history['loss'])
plt.plot(callbacks.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(outputDir+'learningcurve.pdf') 
#plt.show()

plt.figure(2)
plt.plot(callbacks.history.history['acc'])
#print(callbacks.history.history['val_loss'],history.history['loss'])
plt.plot(callbacks.history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(outputDir+'accuracycurve.pdf')

