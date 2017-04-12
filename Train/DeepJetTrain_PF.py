

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


nepochs=100
batchsize=10000
startlearnrate=0.0005
from DeepJet_callbacks import DeepJet_callbacks

callbacks=DeepJet_callbacks(stop_patience=300, 
                            
                            lr_factor=0.5,
                            lr_patience=3, 
                            lr_epsilon=0.003, 
                            lr_cooldown=6, 
                            lr_minimum=0.000001, 
                            
                            outputDir=outputDir)
useweights=False
splittrainandtest=0.8
maxqsize=10 #sufficient



from DataCollection import DataCollection

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
from DeepJet_models import Dense_model_broad, Dense_model_broad_flat
#model = Dense_model_Rec(inputs,traind.getTruthShape()[0],shapes,0.3)
model = Dense_model_broad(inputs,traind.getTruthShape()[0],shapes,0.1)
#model = Dense_model_broad(inputs,traind.getTruthShape()[0],shapes,0.1)
print('compiling')


from keras.optimizers import Adam
adam = Adam(lr=startlearnrate)
model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=['accuracy'])

# This stores the history of the training to e.g. allow to plot the learning curve

#stop when val loss does not decrease anymore


ntrainepoch=traind.getSamplesPerEpoch()
nvalepoch=testd.getSamplesPerEpoch()

testd.isTrain=False
traind.isTrain=True

print('split to '+str(ntrainepoch)+' train samples and '+str(nvalepoch)+' test samples')

print('training')



# the actual training
model.fit_generator(traind.generator() , verbose=1,
        steps_per_epoch=traind.getNBatchesPerEpoch(), 
        epochs=nepochs,
        callbacks=callbacks.callbacks,
        validation_data=testd.generator(),
        validation_steps=testd.getNBatchesPerEpoch(), #)#,
        max_q_size=maxqsize,
        #class_weight = classweights)#,
        class_weight = 'auto')




#######this part should be generarlised!

#options to use are:
print(traind.getUsedTruth())
print(callbacks.history.history.keys())

model.save(outputDir+"KERAS_model.h5")
traind.writeToFile(outputDir+'trainsamples.dc')
testd.writeToFile(outputDir+'valsamples.dc')


# summarize history for loss for trainin and test sample
plt.plot(callbacks.history.history['loss'])
#print(callbacks.history.history['val_loss'],history.history['loss'])
plt.plot(callbacks.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(outputDir+'learningcurve.pdf') 
#plt.show()

plt.figure(2)
plt.plot(callbacks.history.history['acc'])
#print(callbacks.history.history['val_loss'],history.history['loss'])
plt.plot(callbacks.history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(outputDir+'accuracycurve.pdf')

exit()

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
    


