

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

parser = ArgumentParser('Run the training')
parser.add_argument('inputDataDir')
parser.add_argument('outputDir')
args = parser.parse_args()

inputDataDir = os.path.abspath(args.inputDataDir)
inputDataDir+='/'
outputDir=args.outputDir

# create output dir

if os.path.isdir(outputDir):
    print('output directory must not exists yet')
    raise Exception('output directory must not exists yet')

os.mkdir(outputDir)
outputDir = os.path.abspath(outputDir)
outputDir+='/'

#copy configuration to output dir

import shutil
shutil.copyfile(sys.argv[0],outputDir+sys.argv[0])
shutil.copyfile('DeepJet_models.py',outputDir+'DeepJet_models.py')


######################### KERAS PART ######################

# configure the in/out/split etc

testrun=True

nepochs=10
batchsize=10000
learnrate=0.0003#/4
useweights=True
splittrainandtest=0.95
maxqsize=5

from TrainData_deepCSV_ST import TrainData_deepCSV_ST
useDataClass=TrainData_deepCSV_ST




#from from keras.models import Sequential
from DeepJet_models import Dense_model

inputs = Input(shape=(66,))
model = Dense_model(inputs,3)

print('compiling')

from keras.optimizers import Adam
adam = Adam(lr=learnrate)
model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=['accuracy'])

# This stores the history of the training to e.g. allow to plot the learning curve
from keras.callbacks import History # , TensorBoard
# loss per epoch
history = History()


from DataCollection import DataCollection
traind=DataCollection()
traind.readFromFile(inputDataDir+'/dataCollection.dc')
traind.setBatchSize(batchsize)
traind.useweights=useweights

if testrun:
    traind.split(0.02)
    nepochs=2

testd=traind.split(splittrainandtest)

ntrainepoch=traind.getSamplesPerEpoch()
nvalepoch=testd.getSamplesPerEpoch()

print('splitted to '+str(ntrainepoch)+' train samples and '+str(nvalepoch)+' test samples')

print('training')

# the actual training
model.fit_generator(traind.generator(useDataClass()) ,
        samples_per_epoch=ntrainepoch, nb_epoch=nepochs,max_q_size=maxqsize,callbacks=[history],
        validation_data=testd.generator(useDataClass()),
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
plt.title('model accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(outputDir+'accuracycurve.pdf')


features_val=testd.getAllFeatures(TrainData_deepCSV_ST())[0]
labels_val=testd.getAllLabels(TrainData_deepCSV_ST())[0]


predict_test = model.predict(features_val)

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

plt.figure(5)

labels_val_isB= (labels_val[:,0]==1)
# make boolean array for bs, can be used as filter by using it as index (advanced indexing)
predict_test_Bs= predict_test[labels_val_isB]
plt.plot(predict_test_Bs)


labels_val_isL=(labels_val[:,2]==1)
 # assuming 3 classes!, i.e index 2 = used
# make boolean array for bs, can be used as filter by using it as index (advanced indexing)
predict_test_Ls= predict_test[labels_val_isL]
plt.plot(predict_test_Ls)


plt.savefig(outputDir+'name.pdf')

from root_numpy import array2root

# to add back to raw root for more detaiel ROCS and debugging
all_write = np.core.records.fromarrays(  np.hstack((predict_test,labels_val)).transpose(), 
                                             names='probB, probC, probUDSG, isB, isC, isUDSG',
                                             formats = 'float32,float32,float32,float32,float32,float32')
#labels_val
print(all_write.shape)

array2root(all_write,outputDir+"KERAS_result_val.root",mode="recreate")


#from keras.models import load_model
model.save(outputDir+"KERAS_model.h5")
traind.writeToFile(outputDir+'trainsamples.dc')
testd.writeToFile(outputDir+'testsamples.dc')


