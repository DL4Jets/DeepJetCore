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
model = Dense_model(inputs,3)

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
traind.setBatchSize(10000)
traind.useweights=True

testd=traind.split(0.95)

ntrainepoch=traind.getSamplesPerEpoch()
nvalepoch=testd.getSamplesPerEpoch()

print(nvalepoch)

# get sample size from split files and use for the fit_generator function

from TrainData_deepCSV_ST import TrainData_deepCSV_ST


# Tensorbord file (illustrated NN structure)
#TBcallback = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

print('training')

# the actual training
model.fit_generator(traind.generator(TrainData_deepCSV_ST()) ,
        samples_per_epoch=ntrainepoch, nb_epoch=1,max_q_size=5,callbacks=[history],
        validation_data=testd.generator(TrainData_deepCSV_ST()),
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

plt.figure(2)
plt.plot(history.history['acc'])
#print(history.history['val_loss'],history.history['loss'])
plt.plot(history.history['val_acc'])
plt.title('model accurency')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(outputDir+'accurencycurve.pdf')


features_val=testd.getOneFileFeatures(TrainData_deepCSV_ST())
labels_val=testd.getOneFileLabels(TrainData_deepCSV_ST())


predict_test = model.predict(features_val)

# to add back to raw root for more detaiel ROCS and debugging
predict_write = np.core.records.fromarrays(  predict_test.transpose(), 
                                             names='probB, probC, probUDSG',
                                             formats = 'float32,float32,float32')

# this makes you some ROC curves
from sklearn.metrics import roc_curve

# ROC one against all
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

# ROC one against som others
plt.figure(4)
# b vs light (assumes truth C is at index 1 and b truth at 0
labels_val_noC = (labels_val[:,1] == 1)
labels_val_killedC = labels_val[np.invert(labels_val_noC) ]
predict_test_killedC = predict_test[np.invert(labels_val_noC)]
fprC , tprC, _ = roc_curve(labels_val_killedC[:,0], predict_test_killedC[:,0])
BvsL, = plt.plot(tprC, fprC, label='b vs. light')
# b vs c (assumes truth light is at index 2
labels_val_noL = (labels_val[:,2] ==1)

labels_val_killedL = labels_val[np.invert(labels_val_noL)]
predict_test_killedL = predict_test[np.invert(labels_val_noL)]
fpr , tpr, _ = roc_curve(labels_val_killedL[:,0], predict_test_killedL[:,0])
BvsC, = plt.plot(tpr, fpr, label='b vs. c')
plt.semilogy()
#plt.legend([BvsL,BvsC],loc='upper left')
plt.ylabel('BKG efficiency')
plt.xlabel('b efficiency')
plt.ylim((0.001,1))
plt.grid(True)
plt.savefig(outputDir+'ROCs_multi.pdf')

from root_numpy import array2root
array2root(predict_write,outputDir+"KERAS_result_val.root",mode="recreate")


#from keras.models import load_model
model.save(outputDir+"KERAS_model.h5")
traind.writeToFile(outputDir+'trainsamples.dc')
testd.writeToFile(outputDir+'testsamples.dc')


