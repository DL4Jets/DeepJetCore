from __future__ import absolute_import
from __future__ import division
from pdb import set_trace
from argparse import ArgumentParser
# argument parsing and bookkeeping
parser = ArgumentParser('Run the training')
parser.add_argument('mode', choices=['class', 'reg', 'both'], help='What to run (classification, regression, both)')
parser.add_argument('inputfile')
parser.add_argument('outputDir')
parser.add_argument('testfile')
parser.add_argument('-f','--force', action='store_true', help='overwrite the directory if there')
parser.add_argument('--mse', action='store_true', help='use mean-square error as loss')
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
from keras.optimizers import SGD

## to call it from cammand lines
import os
import shutil

from glob import glob
outputDir = args.outputDir
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
   'nepochs'   : 3,
   'batchsize' : 10000,
   'startlearnrate' : 0.0005,
   'useweights' : False,
   'splittrainandtest' : 0.8,
   'maxqsize' : 100, #sufficient
   'conv_dropout' : 0.1,
   'loss_weights' : [1., .25] if not args.mse else [1., .00017],
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
from TrainData_PT_recur import TrainData_PT_recur

traind = DataCollection(args.inputfile)
traind.useweights = config_args['useweights']
    
testd = traind.split(config_args['splittrainandtest'])
input_shapes = traind.getInputShapes()
output_shapes = traind.getTruthShape()

from Losses import loss_NLL
from keras.optimizers import Adam
adam = Adam(lr = config_args['startlearnrate'])

model = None
modifier = None
class_weight = None
def identity(generator):
    for i in generator:
        yield i

if args.mode == 'class':
    model = TrainData_PT_recur.classification_model(input_shapes, output_shapes[0])
    model.compile(
        loss = 'categorical_crossentropy',
        optimizer = adam, metrics = ['categorical_accuracy']
    )
    modifier = TrainData_PT_recur.classification_generator
    class_weight = 'auto'
elif args.mode == 'reg':
    modifier = TrainData_PT_recur.regression_generator
    if args.mse:
        model = TrainData_PT_recur.mse_regression_model(input_shapes)
        model.compile(
            loss = 'mse', optimizer = adam, 
            metrics = ['MSE'],
        )
    else:
        model = TrainData_PT_recur.regression_model(input_shapes)
        model.compile(
            loss = loss_NLL, optimizer = adam, 
            metrics = ['MSE'],
        )
else:
    modifier = identity
    if args.mse:
        model = TrainData_PT_recur.mse_model(input_shapes, output_shapes[0])
        model.compile(
            loss = ['categorical_crossentropy', 'MSE'], #apply xentropy to the first output (flavour) and NLL to the pt regression
            optimizer = adam, metrics = ['categorical_accuracy','MSE'],
            loss_weights = config_args['loss_weights']
        )
    else:
        model = TrainData_PT_recur.model(input_shapes, output_shapes[0])
        model.compile(
            loss = ['categorical_crossentropy', loss_NLL], #apply xentropy to the first output (flavour) and NLL to the pt regression
            optimizer = adam, metrics = ['categorical_accuracy','MSE'],
            loss_weights = config_args['loss_weights']
        )

ntrainepoch = traind.getSamplesPerEpoch()
nvalepoch   = testd.getSamplesPerEpoch()

testd.isTrain  = False
traind.isTrain = True

print 'split to %d train samples and %d test samples' % (ntrainepoch, nvalepoch)
#for bookkeeping
traind.setBatchSize(config_args['batchsize'])
testd.setBatchSize(500)
traind.writeToFile(outputDir+'trainsamples.dc')
testd.writeToFile( outputDir+'valsamples.dc')

print 'training'
try:
    model.fit_generator(
        modifier(traind.generator()),
        verbose=1,
        steps_per_epoch = traind.getNBatchesPerEpoch(), 
        epochs = config_args['nepochs'],
        callbacks = callbacks.callbacks,
        validation_data = modifier(testd.generator()),
        validation_steps = testd.getNBatchesPerEpoch(),
        max_q_size = config_args['maxqsize'], #maximum size for the generator queue
        class_weight = class_weight,
    )
except KeyboardInterrupt:
    print "\n\nEarly manual stopping received!\n\n"
except Exception as d:
    raise d

####################################################
#                                                  #
#           Plots and jsons to make them           #
#                                                  #
####################################################

model.save(outputDir+"KERAS_model.h5")

# summarize history for loss for trainin and test sample
plt.plot(callbacks.history.history['loss'])
#print(callbacks.history.history['val_loss'],history.history['loss'])
plt.plot(callbacks.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(outputDir+'learningcurve.pdf') 
plt.clf()
#plt.show()

import json
def normalize(inmap):
    ret = {}
    for i in inmap:
        ret[i] = [float(j) for j in inmap[i]]
    return ret
                  
with open(outputDir+'history.json', 'w') as history:
    history.write(json.dumps(normalize(callbacks.history.history)))

plt.plot(*callbacks.timer.points)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('time [s]')
plt.savefig(outputDir+'loss_vs_time.pdf')
plt.clf()

with open(outputDir+'loss_vs_time.json', 'w') as timeloss:
    jmap = {
        'elapsed' : callbacks.timer.points[0],
        'loss' : callbacks.timer.points[1]
    }
    timeloss.write(json.dumps(normalize(jmap)))


####################################################
#                                                  #
#         Testing, because it takes ages           #
#                                                  #
####################################################

from subprocess import Popen, PIPE

parser.add_argument('inputfile')
parser.add_argument('outputDir')
parser.add_argument('testfile')
cmd = [
    'predict.py', 
    '%s/KERAS_check_best_model.h5' % args.outputDir,
    args.testfile,
    '%s/testing' % args.outputDir
    ]
if (args.mode == 'reg' or args.mode == 'both') and not args.mse:
    cmd.append('--NLLloss')

print 'Executing command:'
print ' '.join(cmd)
proc = Popen(cmd, stdout=PIPE, stderr=PIPE)
out, err = proc.communicate()
code = proc.wait()

if code:
    raise RuntimeError('Command failed with error message: \n\n%s\n\n' % err)
