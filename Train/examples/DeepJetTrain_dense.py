from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# some private extra plots
#from  NBatchLogger import NBatchLogger




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
inputDataDirVal = sys.argv[3]
os.mkdir(outputDir)
import shutil
shutil.copyfile(sys.argv[0],outputDir+sys.argv[0])
shutil.copyfile('DeepJet_models.py',outputDir+'DeepJet_models.py')

# here we read the data
# the old ntupels for comparison
#features = np.load(inputDataDir+'global_X_2016.npy',mmap_mode='r')

# this is the old sample

# The new sample is
#features= np.load(inputDataDir+'global_X_2016.npy')
features= np.load(inputDataDir+'global_X.npy')
features_val   = np.load(inputDataDirVal+'global_X.npy')
#features_val     = np.load('/afs/cern.ch/work/m/mstoye/root_numpy/debugTrack/JetTaggingVariables_ttbar_prepro_X.npy')
#features_val =  np.delete(features_val, [2,3,4,5], 1)
#features = np.memmap(inputDataDir+'global_X_2016.npy', dtype='float32', mode='r',shape=(38156556, 66))
#features = np.delete(features, [2,3,4,5], 1)
#labels = np.load(inputDataDir+'MIX_Y.npy')
# using view would be quicker but longer syntax

# this is the old 2016 sample
#labels = np.load('/afs/cern.ch/work/m/mstoye/root_numpy/debugTrack/MIX_Y.npy')
#labels = np.load(inputDataDir+'classtruth_2016.npy',mmap_mode='r')
# The new sample is
#labels = np.load(inputDataDir+'classtruth_2016.npy')
labels = np.load(inputDataDir+'class_truth3.npy')
labels_val = np.load(inputDataDirVal+'class_truth3.npy')

# this gives the true 2016 sample
#labels_val   = np.load('/afs/cern.ch/work/m/mstoye/root_numpy/debugTrack/JetTaggingVariables_ttbar_clean_Y3.npy')
#labels_val =labels_val.transpose()

#labels = np.array(labels.tolist())
weights = np.load(inputDataDir+'weights.npy')
inputs = Input(shape=(66,))

#from from keras.models import Sequential
from DeepJet_models import Dense_model
model = Dense_model(inputs)

from keras.optimizers import Adam
adam = Adam(lr=0.0003)
#sgd = SGD(lr=0.03)
#model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=['accuracy'])

# This stores the history of the training to e.g. allow to plot the learning curve
from keras.callbacks import History # , TensorBoard
# loss per epoch
history = History()

# Tensorbord file (illustrated NN structure)
#TBcallback = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

# the actual training
model.fit(features, labels,validation_data=(features_val,labels_val), nb_epoch=25, batch_size=10000, callbacks=[history])#, sample_weight=weights)
#model.fit_generator(datagen.flow(features, labels,batch_size=50000),
# samples_per_epoch=features.shape[0],
#                    nb_epoch=10)


import matplotlib
#if no X11 use below
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# summarize history for loss for trainin and test sample
plt.figure(1)
plt.plot(history.history['loss'])
#print(history.history['val_loss'],history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(outputDir+'learningcurve.pdf') 
#plt.show()
plt.figure(2)
plt.plot(history.history['acc'])
#print(history.history['val_loss'],history.history['loss'])
plt.plot(history.history['val_acc'])
plt.title('model accurency')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(outputDir+'accurencycurve.pdf')

predict_test = model.predict(features_val)

# to add back to raw root for more detaiel ROCS and debugging
predict_write = np.core.records.fromarrays(  predict_test.transpose(), 
                                             names='probB, probC, probUDSG',
                                             formats = 'float32,float32,float32')

# this makes you some ROC curves
from sklearn.metrics import roc_curve
plt.figure(3)
for i in range(labels_val.shape[1]):
#    print (i , ' is', labels_val[i][:], ' ', predict_test[i][:])
    
    fpr , tpr, _ = roc_curve(labels_val[:,i], predict_test[:,i])
 #   print (fpr, ' ', tpr, ' ', _)
    plt.plot(tpr, fpr, label=predict_write.dtype.names[i])
print (predict_write.dtype.names)
plt.semilogy()
plt.legend(predict_write.dtype.names, loc='upper left')
plt.savefig(outputDir+'ROCs.pdf')

from root_numpy import array2root
array2root(predict_write,outputDir+"KERAS_result_val.root",mode="recreate")
from keras.models import load_model
model.save(outputDir+"KERAS_model.h5")
