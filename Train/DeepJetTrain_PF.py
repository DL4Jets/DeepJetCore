

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

from DeepJet_models import Dense_model,Dense_model2, Dense_model_broad,Dense_model_broad_flat
from TrainData_deepCSV_ST import TrainData_deepCSV_ST


def predictAndMakeRoc(features_val, labels_val, nameprefix, names,formats, model):

    import numpy

    predict_test = model.predict(features_val)
    metric=model.evaluate(features_val, labels_val, batch_size=10000)
    
    print(metric)
    
    predict_write = np.core.records.fromarrays(  predict_test.transpose(), 
                                                 names=names,
                                                 formats = formats)
    
    # this makes you some ROC curves
    from sklearn.metrics import roc_curve
    labels_val = numpy.array(labels_val[0])
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

nepochs=500
batchsize=10000
startlearnrate=0.003
useweights=False
splittrainandtest=0.75
maxqsize=10 #sufficient



from DataCollection import DataCollection
from TrainData_deepCSV_ST import TrainData_deepCSV_ST
from TrainData_deepCSV import TrainData_deepCSV

traind=DataCollection()
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

print(traind.getTruthShape()[0])

#from from keras.models import Sequential

from keras.layers import Input
inputs = [Input(shape=shapes[0]),
          Input(shape=shapes[1]),
          Input(shape=shapes[2]),
          Input(shape=shapes[3])]

#model = Dense_model2(inputs,traind.getTruthShape()[0],(traind.getInputShapes()[0],))

print(traind.getTruthShape()[0])
model = Dense_model_broad(inputs,traind.getTruthShape()[0],shapes,0.2)
print('compiling')

from keras.optimizers import Adam
adam = Adam(lr=startlearnrate)
model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=['accuracy'])

# This stores the history of the training to e.g. allow to plot the learning curve
from keras.callbacks import History, LearningRateScheduler, EarlyStopping #, ReduceLROnPlateau # , TensorBoard
# loss per epoch
history = History()

#stop when val loss does not decrease anymore
stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')

from ReduceLROnPlateau import ReduceLROnPlateau


LR_onplatCB = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, 
                                mode='auto', verbose=1, epsilon=0.001, cooldown=0, min_lr=0.00001)



ntrainepoch=traind.getSamplesPerEpoch()
nvalepoch=testd.getSamplesPerEpoch()

testd.isTrain=False
traind.isTrain=True

print('split to '+str(ntrainepoch)+' train samples and '+str(nvalepoch)+' test samples')

print('training')



# the actual training
model.fit_generator(traind.generator() ,
        samples_per_epoch=ntrainepoch, nb_epoch=nepochs,max_q_size=maxqsize,callbacks=[history,stopping,LR_onplatCB],
        validation_data=testd.generator(),
        nb_val_samples=nvalepoch, #)#,
        #class_weight = classweights)#,
        class_weight = 'auto')



#######this part should be generarlised!

#options to use are:
print(traind.getUsedTruth())
print(history.history.keys())

model.save(outputDir+"KERAS_model.h5")
traind.writeToFile(outputDir+'trainsamples.dc')
testd.writeToFile(outputDir+'testsamples.dc')


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

features_val=testd.getAllFeatures()
labels_val=testd.getAllLabels()

print('making rocs',len(labels_val))

weights_val=testd.getAllWeights()[0]


names='probB, probC, probUDSG'
formats='float32,float32,float32'
predictAndMakeRoc(features_val, labels_val, outputDir+"all_val", names,formats,model)
labelsandweights = labels_val[0] #np.concatenate((labels_val,weights_val.T),axis=1)

from root_numpy import array2root

predict_test = model.predict(features_val)
# to add back to raw root for more detaiel ROCS and debugging
all_write = np.core.records.fromarrays(  np.hstack((predict_test,labelsandweights)).transpose(), 
                                             names='probB, probC, probUDSG, isB, isC',# isUDSG,weights',
                                             formats = 'float32,float32,float32,float32,float32,float32')#,float32')
#labels_val
print(all_write.shape)

array2root(all_write,outputDir+"KERAS_result_val.root",mode="recreate")


#from keras.models import load_model


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
    


