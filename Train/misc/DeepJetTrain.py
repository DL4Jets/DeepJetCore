from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# some private extra plots
#from  NBatchLogger import NBatchLogger

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K 

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, merge, Input
#zero padding done before
#from keras.layers.convolutional import Cropping1D, ZeroPadding1D
from keras.optimizers import SGD

y = np.load('truth.npy')
x = np.load('PFtestIII.npy')

from keras.models import Model
def Sequential_model(input_shape):
    model = Sequential()
    model.add(Convolution2D(3, 1,1, border_mode='same', activation='relu',init='lecun_uniform',input_shape=input_shape))
#model.add(Dense(10, activation='linear',init='normal',input_shape=(10,30)))
    model.add(Flatten())
    #model.add(Dropout(0.25))
    model.add(Dense(1, activation='linear',init='normal'))
    return model
# more on https://keras.io/getting-started/functional-api-guide/

if len(x.shape) != 4:
    print ('macro intended for 2D convolutions! Input must be 4D')
input_shape = (x.shape[1],x.shape[2],x.shape[3])
model = Sequential_model(input_shape)

sgd = SGD()
from keras.optimizers import Adam
adam = Adam(lr=0.03)
model.compile(loss='mean_squared_error', optimizer=adam)
# This stores the history of the training to e.g. allow to plot the learning curve
from keras.callbacks import History # , TensorBoard
# loss per epoch
history = History()

# loss per batch
#nBatchLogger = NBatchLogger(80)

# Tensorbord file (illustrated NN structure)
#TBcallback = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

# the actual training
model.fit(x, y ,validation_split=0.99, nb_epoch=5, batch_size=1000, callbacks=[history])

#The below plots the loss curve for the batches, this is higher granularity than the history (per epoch)
#plt.plot(nBatchLogger.losses)
#plt.plot(nBatchLogger.val_losses)
#plt.ylabel('loss')
#plt.xlabel('n batches')
#plt.legend(['val', 'train'], loc='upper left')
#plt.show()


# summarize history for loss for trainin and test sample
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
##plt.savefig('dense10.pdf') 
plt.show()

