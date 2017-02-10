from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# some private extra plots
#from  NBatchLogger import NBatchLogger

import matplotlib
#if no X11 use below
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K 

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, merge, Input
#zero padding done before
#from keras.layers.convolutional import Cropping1D, ZeroPadding1D
from keras.optimizers import SGD

reg_truth = np.load('regres_truth.npy')
x_local = np.load('local_X.npy')
x_global = np.load('global_X.npy')
class_truth = np.load('class_truth.npy')
# using view would be quicker but longer syntax
class_truth = np.array(class_truth.tolist())
weights = np.load('weights.npy')

print('shapes ' , reg_truth.shape, ' ' , x_local.shape , '  x_global ' , x_global.shape,' class_truth ',class_truth.shape)
print(class_truth[0][2])
from keras.models import Model

def Incept_model(inputs,dropoutRate=0.1):
    """
    This NN adds two inputs, one for a conv net and a seceond for a dense net, both nets get combined. The last layer is split into regression and classification activations (softmax, linear)
    """
    
    x =   Convolution2D(50, 1 , 1, border_mode='same', activation='relu',init='lecun_uniform',input_shape=(10,30,2))(inputs[0])
    # add more layers to get deeper
    x =   Convolution2D(50, 1 , 1, border_mode='same', activation='relu',init='lecun_uniform')(x)
    x =   Convolution2D(10, 1 , 1, border_mode='same', activation='relu',init='lecun_uniform')(x)
    x = Flatten()(x)
    #  Here add e.g. the normal dense stuff from DeepCSV
    y = Dense(1, activation='relu',init='lecun_uniform',input_shape=(1,))(inputs[1])
    y = Dense(1, activation='relu',init='lecun_uniform')(y)
    y = Dense(1, activation='relu',init='lecun_uniform')(y)
    # add more layers to get deeper
    
    # combine convolutional and dense (global) layers
    x = merge( [x,y ] , mode='concat')

    # linear activation for regression and softmax for classification
    x = Dense(100, activation='relu',init='normal')(x)
    #x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',init='normal')(x)
#    x = Dropout(dropoutRate)(x)
    x=  Dense(100, activation='relu',init='normal')(x)
#    x = Dropout(dropoutRate)(x)
    # add more layers to get deeper

    predictions = [Dense(1, activation='linear',init='normal')(x),Dense(4, activation='softmax',init='lecun_uniform')(x)]
    model = Model(input=inputs, output=predictions)
    return model

inputs = [Input(shape=(4,4,332)),Input(shape=(1,))]
model = Incept_model(inputs)

sgd = SGD()
from keras.optimizers import Adam
adam = Adam(lr=0.03)
model.compile(loss=['mean_squared_error','categorical_crossentropy'], optimizer=adam)

# This stores the history of the training to e.g. allow to plot the learning curve
from keras.callbacks import History # , TensorBoard
# loss per epoch
history = History()

# loss per batch
#nBatchLogger = NBatchLogger(80)

# Tensorbord file (illustrated NN structure)
#TBcallback = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

# the actual training
model.fit([x_local, x_global], [reg_truth,class_truth] ,validation_split=0.99, nb_epoch=100, batch_size=1000, callbacks=[history], sample_weight=[weights,weights])

#The below plots the loss curve for the batches, this is higher granularity than the history (per epoch)
#plt.plot(nBatchLogger.losses)
#plt.plot(nBatchLogger.val_losses)
#plt.ylabel('loss')
#plt.xlabel('n batches')
#plt.legend(['val', 'train'], loc='upper left')
#plt.show()


# summarize history for loss for trainin and test sample
plt.plot(history.history['loss'])
print(history.history['val_loss'],history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
##plt.savefig('dense10.pdf') 
plt.show()

