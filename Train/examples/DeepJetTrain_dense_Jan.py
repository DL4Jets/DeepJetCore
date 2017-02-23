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

inputDataDir = sys.argv[1]
if inputDataDir[-1] != "/":
    inputDataDir+="/"
outputFilesTag = sys.argv[2]
outputDir = inputDataDir+outputFilesTag+"/"
os.mkdir(outputDir)
import shutil
shutil.copyfile(sys.argv[0],outputDir+sys.argv[0])
shutil.copyfile('DeepJet_models.py',outputDir+'DeepJet_models.py')

inputs = Input(shape=(66,))

#from from keras.models import Sequential
from DeepJet_models import Dense_model
model = Dense_model(inputs)

print('compiling')


from keras.optimizers import Adam
adam = Adam(lr=0.0003)
model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=['accuracy'])

# This stores the history of the training to e.g. allow to plot the learning curve
from keras.callbacks import History # , TensorBoard
# loss per epoch
history = History()

print('splitting')

from DataCollection import DataCollection
traind=DataCollection()
traind.readFromFile(inputDataDir+'/dataCollection.dc')
traind.setBatchSize(1000)
traind.useweights=True

testd=traind.split(0.95)

ntrainepoch=traind.getSamplesPerEpoch()
nvalepoch=testd.getSamplesPerEpoch()

print(nvalepoch)

# get sample size from split files and use for the fit_generator function

from TrainData_deepCSV import TrainData_deepCSV


# Tensorbord file (illustrated NN structure)
#TBcallback = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

print('training')

# the actual training
model.fit_generator(traind.generator(TrainData_deepCSV()) ,
        samples_per_epoch=ntrainepoch, nb_epoch=1,max_q_size=5,callbacks=[history],
        validation_data=testd.generator(TrainData_deepCSV()),
        nb_val_samples=nvalepoch)#, sample_weight=weights)
#model.fit_generator(datagen.flow(features, labels,batch_size=50000),
# samples_per_epoch=features.shape[0],
#                    nb_epoch=10)


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

#from keras.models import load_model
model.save(outputDir+"KERAS_model.h5")
traind.writeToFile(outputDir+'trainsamples.dc')
testd.writeToFile(outputDir+'testsamples.dc')


