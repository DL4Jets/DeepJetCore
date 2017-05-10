from __future__ import absolute_import
from __future__ import division

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
from argparse import ArgumentParser
import shutil
from pdb import set_trace

# argument parsing and bookkeeping
parser = ArgumentParser('Run the training')
parser.add_argument('inputDataCollection')
parser.add_argument('outputDir')
parser.add_argument('-f','--force', action='store_true', help='overwrite the directory if there')
args = parser.parse_args()

inputData = os.path.abspath(args.inputDataCollection)
outputDir=args.outputDir
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
   'nepochs'   : 1,
   'batchsize' : 500,
   'startlearnrate' : 0.0005,
   'useweights' : False,
   'splittrainandtest' : 0.8,
   'maxqsize' : 10, #sufficient
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

traind = DataCollection()
traind.readFromFile(inputData)
traind.setBatchSize(config_args['batchsize'])
traind.useweights=config_args['useweights']

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
#  To be moved to a function in DeepJet_models.py  #
#                                                  #
####################################################

from keras.layers import Dense, Dropout, Flatten, Convolution2D, concatenate, \
   Convolution1D, Convolution3D, Reshape, LocallyConnected2D
from keras.models import Model

#unpack inputs
glob, charged, neutral, svs, bin_global = inputs

def make_layers(layer_type, args_list, dropout, 
                layer_in, dropout_at_first=False, **kwargs):
    ret = layer_in
    first = not dropout_at_first
    for args in args_list:
        if first:
            first = False
        else:
            ret = Dropout(dropout)(ret)
        ret = layer_type(*args, **kwargs)(ret)
    return ret    

#build single bin convolutions
kwargs  = {'kernel_initializer' : 'lecun_uniform',  'activation' : 'relu'}
k = (1,1,1) #kernel
charged = make_layers(Convolution3D, [[64, k], [32, k], [32, k], [8, k]], config_args['conv_dropout'], charged, **kwargs)
svs     = make_layers(Convolution3D, [[64, k], [32, k], [32, k], [8, k]], config_args['conv_dropout'], svs, **kwargs)
neutral = make_layers(Convolution3D, [[32, k], [16, k], [4 , k]], config_args['conv_dropout'], neutral, **kwargs)

#flatten the single bins
charged = Reshape((
      int(charged.shape[1]), # [0] is the number of batches, set to None, 1 is the # of x bins, returns Dimension(5), we need to cast to int
      int(charged.shape[2]), # 2 is the # of y bins
      int(charged.shape[3]*charged.shape[4]),
      ))(charged)
neutral = Reshape((
      int(neutral.shape[1]), 
      int(neutral.shape[2]), 
      int(neutral.shape[3]*neutral.shape[4]),
      ))(neutral)
svs = Reshape((
      int(svs.shape[1]), 
      int(svs.shape[2]), 
      int(svs.shape[3]*svs.shape[4]),
      ))(svs)

for ib in [1,2]:
   if not (charged.shape[ib] == neutral.shape[ib] == svs.shape[ib]):
      raise ValueError('The number of bins along the axis %d should be consistent for charged, neutral and sv features' % ib)

#merge the info from different sources, but from the same bin, into a single place
binned_info = concatenate([charged,neutral,svs,bin_global]) #shape (?, 10, 10, 76)

#shrink the info size
nentries_per_bin = int(binned_info.shape[-1])
k = (1,1)
binned_info = make_layers(
    Convolution2D, [[nentries_per_bin, k], [25, k]], 
    config_args['conv_dropout'], binned_info, 
    dropout_at_first=True, **kwargs
    )

#learn a different representation for each bin
#and reduce the dimensionality to 3x3
xbins, ybins = int(binned_info.shape[1]), int(binned_info.shape[2])
kernel = xbins//3
if xbins != ybins:
    raise ValueError('The number of x and y bins should be the same!')
if xbins % 3 != 0:
    raise ValueError('The number of x and y bins should be a multiplier of 3')

binned_info = make_layers(
    LocallyConnected2D, [[40, kernel]], 
    config_args['conv_dropout'], binned_info,
    dropout_at_first=True, strides=kernel,
    data_format='channels_last', kernel_initializer='lecun_uniform',
    activation='relu'
)

#
# Dense part
#

binned_info = Reshape((
      int(binned_info.shape[1]) * 
      int(binned_info.shape[2]) * 
      int(binned_info.shape[3]), #coma to make it a tuple
      ))(binned_info) #Flatten()(binned_info) seems to fuck up the tensor output shape, so I reshape
X = concatenate([glob, binned_info])

X = make_layers(
    Dense, [[int(X.shape[1])]]+[[100]]*7, 
    config_args['conv_dropout'], X,
    dropout_at_first=True, kernel_initializer='lecun_uniform',
    activation='relu'
)
X = Dropout(config_args['conv_dropout'])(X)

# regression
pt_sigma = Dense(2, activation='linear', kernel_initializer='lecun_uniform')(X)
# classification
flavour = Dense(output_shapes[0], activation='softmax',kernel_initializer='lecun_uniform')(X)
model = Model(inputs=list(inputs), outputs=[flavour, pt_sigma])


####################################################
#                                                  #
#         Compiling model and setting stuff        #
#                                                  #
####################################################

print 'compiling'

def loss_NLL(y_true, x):
    x_pred = x[:,1:]
    x_sig = x[:,:1]
    return K.mean(0.5* K.log(K.square(x_sig))  + K.square(x_pred - y_true)/K.square(x_sig)/2.,    axis=-1)


from keras.optimizers import Adam
adam = Adam(lr = config_args['startlearnrate'])
model.compile(
   loss = ['categorical_crossentropy', loss_NLL], #apply xentropy to the first output (flavour) and NLL to the pt regression
   optimizer = adam, metrics = ['accuracy','accuracy'],
   loss_weights = config_args['loss_weights']
   )

ntrainepoch = traind.getSamplesPerEpoch()
nvalepoch   = testd.getSamplesPerEpoch()

testd.isTrain  = False
traind.isTrain = True

print 'split to %d train samples and %d test samples' % (ntrainepoch, nvalepoch)
#for bookkeeping
traind.writeToFile(outputDir+'trainsamples.dc')
testd.writeToFile( outputDir+'valsamples.dc')

print 'training'

####################################################
#                                                  #
#                The real training                 #
#                                                  #
####################################################

model.fit_generator(
   traind.generator() , verbose=1,
   steps_per_epoch = traind.getNBatchesPerEpoch(), 
   epochs = config_args['nepochs'],
   callbacks = callbacks.callbacks,
   validation_data = testd.generator(),
   validation_steps = testd.getNBatchesPerEpoch(),
   max_q_size = config_args['maxqsize'], #maximum size for the generator queue
   class_weight = ['auto', 'auto']
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
plt.plot(callbacks.history.history['dense_10_acc'])
#print(callbacks.history.history['val_loss'],history.history['loss'])
plt.plot(callbacks.history.history['val_dense_10_acc'])
plt.title('model accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(outputDir+'accuracycurve.pdf')
