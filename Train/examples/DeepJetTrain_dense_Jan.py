

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
from DeepJet_models import Dense_model
from TrainData_deepCSV_ST import TrainData_deepCSV_ST


def predictAndMakeRoc(features_val, labels_val, nameprefix, names,formats, model):



    predict_test = model.predict(features_val)
    metric=model.evaluate(features_val, labels_val, batch_size=10000)
    
    print(metric)
    
    predict_write = np.core.records.fromarrays(  predict_test.transpose(), 
                                                 names=names,
                                                 formats = formats)
    
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
    plt.savefig(nameprefix+'ROCs.pdf')
    plt.close(3)
    
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
    plt.savefig(nameprefix+'ROCs_multi.pdf')
    plt.close(4)
    
    return metric
    
# argument parsing and bookkeeping

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


######################### KERAS PART ######################

# configure the in/out/split etc

testrun=False

nepochs=5
batchsize=10000
learnrate=0.0003#/4
useweights=False
splittrainandtest=0.9
maxqsize=1 #sufficient

useDataClass=TrainData_deepCSV_ST




#from from keras.models import Sequential

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
traind.readFromFile(inputData)
traind.setBatchSize(batchsize)
traind.useweights=useweights

#does this cause the weird behaviour?
#traind.removeLast()

if testrun:
    traind.split(0.02)
    nepochs=2


testd=traind.split(splittrainandtest)

#tmp=traind
#traind=testd
#testd=tmp

testd.isTrain=False

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
names='probB, probC, probUDSG'
formats='float32,float32,float32'
predictAndMakeRoc(features_val, labels_val, outputDir+"all_val", names,formats,model)


from root_numpy import array2root

predict_test = model.predict(features_val)
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


# per file plots. Take lot of time
exit()

metrics=[]
print('making individual ROCs for test data')
for samplefile in testd.samples:
    tdt=useDataClass()
    tdt.readIn(testd.getSamplePath(samplefile))
    print(samplefile)
    metrics.append(predictAndMakeRoc(tdt.x[0],tdt.y[0],outputDir+samplefile+"_val",names,formats,model))
    

print('making individual ROCs for train data')
for samplefile in traind.samples:
    tdt=useDataClass()
    tdt.readIn(traind.getSamplePath(samplefile))
    print(samplefile)
    metrics.append(predictAndMakeRoc(tdt.x[0],tdt.y[0],outputDir+samplefile+"_train",names,formats,model))
    
metricsloss=[]
metricsacc=[]
count=range(0,len(metrics))
for m in metrics:
    metricsloss.append(m[0])
    metricsacc.append(m[1])
    
    

plt.figure(6)
plt.plot(count,metricsloss)
plt.grid(True)
plt.savefig(outputDir+'lossperfile.pdf')
plt.figure(7)
plt.plot(count,metricsacc)
plt.grid(True)
plt.savefig(outputDir+'accperfile.pdf')
    


