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
shutil.copyfile("DeepJetTrain_regr_class_dense_conv.py",outputDir+"DeepJetTrain_regr_class_dense_conv.py")

reg_truth = np.load(inputDataDir+'regres_truth.npy')
x_local = np.load(inputDataDir+'local_X.npy')
x_global = np.load(inputDataDir+'global_X.npy')
x_global = np.array( x_global.tolist() )
class_truth = np.load(inputDataDir+'class_truth.npy')
# using view would be quicker but longer syntax
class_truth = np.array(class_truth.tolist())
weights = np.load(inputDataDir+'weights.npy')
from keras.layers import Input
inputs = [Input(shape=(6,5,122)),Input(shape=(5,))]


from DeepJet_models import Incept_model
model = Incept_model(inputs)

sgd = SGD()
from keras.optimizers import Adam
adam = Adam(lr=0.005)
model.compile(loss=['mean_squared_error','categorical_crossentropy'], optimizer=adam,loss_weights=[.005, 10.])

# This stores the history of the training to e.g. allow to plot the learning curve
from keras.callbacks import History # , TensorBoard
# loss per epoch
history = History()


# the actual training
model.fit([x_local, x_global], [reg_truth,class_truth] ,validation_split=0.99, nb_epoch=20, batch_size=1000, callbacks=[history], sample_weight=[weights,weights])

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
