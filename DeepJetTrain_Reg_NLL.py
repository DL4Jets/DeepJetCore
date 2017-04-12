from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# some private extra plots
#from  NBatchLogger import NBatchLogger

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

from DeepJet_models import Dense_model_reg, Dense_model_broad_reg
#from TrainData_deepCSV_ST import TrainData_deepCSV_ST

# argument parsing and bookkeeping

def loss_NLL(y_true, x):
    x_pred = x[:,1:]
    x_sig = x[:,:1]
    return K.mean(0.5* K.log(K.square(x_sig))  + K.square(x_pred - y_true)/K.square(x_sig)/2.,    axis=-1)

parser = ArgumentParser('Run the training')
parser.add_argument('inputDataCollection')
parser.add_argument('outputDir')
args = parser.parse_args()

inputData = os.path.abspath(args.inputDataCollection)
outputDir=args.outputDir
# create output dir

if os.path.isdir(outputDir):
    print('output directory must not exists yet')
    raise Exception('output directory must not exists yet')

os.mkdir(outputDir)
outputDir = os.path.abspath(outputDir)
outputDir+='/'

#copy configuration to output dir

shutil.copyfile(sys.argv[0],outputDir+sys.argv[0])
shutil.copyfile('../modules/DeepJet_models.py',outputDir+'DeepJet_models.py')


print ('start')


######################### KERAS PART ######################
# configure the in/out/split etc

testrun=False

nepochs=20
batchsize=10000
startlearnrate=0.009
useweights=False
splittrainandtest=0.8
maxqsize=10 #sufficient

from DataCollection import DataCollection
#from TrainData_deepCSV_ST import TrainData_deepCSV_ST
from TrainData_deepCSV import TrainData_deepCSV

traind=DataCollection()
print (inputData, ' shapes ')
traind.readFromFile(inputData)
traind.setBatchSize(batchsize)
traind.useweights=useweights

if testrun:
    traind.split(0.02)
    nepochs=10
    
testd=traind.split(splittrainandtest)
shapes=traind.getInputShapes()

#shapes=[]
#for s in shapesin:
#    _sl=[]
#    for i in range(len(s)):
#        if i:
#            _sl.append(s[i])
#    s=(_sl)
#    shapes.append(s)
#    print(s)
#        

print(shapes)

print(traind.getTruthShape())

#from from keras.models import Sequential

from keras.layers import Input

inputs = [Input(shape=shapes[0]),Input(shape=shapes[1]),Input(shape=shapes[2]),Input(shape=shapes[3]),Input(shape=shapes[4])]

#model = Dense_model2(inputs,traind.getTruthShape()[0],(traind.getInputShapes()[0],))

print(traind.getTruthShape()[0])
model =  Dense_model_broad_reg(inputs,traind.getTruthShape()[0],shapes,0.1,2)
print('compiling')


from keras.optimizers import Adam
adam = Adam(lr=startlearnrate)
model.compile(loss=loss_NLL, 
              optimizer=adam,
             # metrics=['accuracy','accuracy'],
              )#strong focus on flavour

# This stores the history of the training to e.g. allow to plot the learning curve
from keras.callbacks import Callback,History, LearningRateScheduler, EarlyStopping, LambdaCallback,ModelCheckpoint #, ReduceLROnPlateau # , TensorBoard
# loss per epoch
history = History()

#stop when val loss does not decrease anymore
stopping = EarlyStopping(monitor='val_loss', patience=500, verbose=1, mode='min')

from ReduceLROnPlateau import ReduceLROnPlateau


LR_onplatCB = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, 
                                mode='min', verbose=1, epsilon=0.001, cooldown=4, min_lr=0.00001)

modelcheck=ModelCheckpoint(outputDir+"KERAS_check_model.h5", monitor='val_loss', verbose=1, save_best_only=True)

class newline_callbacks(Callback):
    def on_epoch_end(self,epoch, epoch_logs={}):
        print('\n***callsbacks***\n')
        
class newline_callbackss(Callback):
    def on_epoch_end(self,epoch, epoch_logs={}):
        print('\n***callsbacks nd***\n')

nLcb=newline_callbacks()
nLcbe=newline_callbackss()

ntrainepoch=traind.getSamplesPerEpoch()
nvalepoch=testd.getSamplesPerEpoch()

testd.isTrain=False
traind.isTrain=True

print('split to '+str(ntrainepoch)+' train samples and '+str(nvalepoch)+' test samples')

print('training')



# the actual training
model.fit_generator(traind.generator() , verbose=1,
        steps_per_epoch=traind.getNBatchesPerEpoch(), 
        epochs=nepochs,
        callbacks=[nLcb,history,stopping,LR_onplatCB,modelcheck,nLcbe],
        validation_data=testd.generator(),
        validation_steps=testd.getNBatchesPerEpoch(), #)#,
        max_q_size=maxqsize,
        #class_weight = classweights)#,
#        class_weight = 'auto'
        )




#######this part should be generarlised!

#options to use are:
print(traind.getUsedTruth())
print(history.history.keys())

model.save(outputDir+"KERAS_model.h5")
traind.writeToFile(outputDir+'trainsamples.dc')
testd.writeToFile(outputDir+'valsamples.dc')


# summarize history for loss for trainin and test sample
plt.plot(history.history['loss'])
#print(history.history['val_loss'],history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(outputDir+'learningcurve.pdf') 
#plt.show()

