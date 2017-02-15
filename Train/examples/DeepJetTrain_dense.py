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

# here we read the data
features = np.load(inputDataDir+'global_X.npy')
features = np.array( features.tolist() )
labels = np.load(inputDataDir+'class_truth.npy')
# using view would be quicker but longer syntax
labels = np.array(labels.tolist())
weights = np.load(inputDataDir+'weights.npy')
inputs = Input(shape=(5,))

#from from keras.models import Sequential
from DeepJet_models import Dense_model
model = Dense_model(inputs)

from keras.optimizers import Adam
adam = Adam(lr=0.005)
model.compile(loss='categorical_crossentropy', optimizer=adam)

# This stores the history of the training to e.g. allow to plot the learning curve
from keras.callbacks import History # , TensorBoard
# loss per epoch
history = History()

# Tensorbord file (illustrated NN structure)
#TBcallback = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

# the actual training
model.fit(features, labels,validation_split=0.90, nb_epoch=20, batch_size=1000, callbacks=[history], sample_weight=weights)

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

from keras.models import load_model
model.save(outputDir+"KERAS_model.h5")
