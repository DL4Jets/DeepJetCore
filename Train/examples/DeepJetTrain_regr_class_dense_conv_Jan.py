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




from DataCollection import DataCollection
from TrainData_veryDeepJet import TrainData_veryDeepJet

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

#os.mkdir(outputDir)
#import shutil
#shutil.copyfile("DeepJetTrain_regr_class_dense_conv.py",outputDir+"DeepJetTrain_regr_class_dense_conv.py")



from keras.layers import Input
inputs = [Input(shape=(6,5,122)),Input(shape=(5,))]


from DeepJet_models import Incept_model
model = Incept_model(inputs)

sgd = SGD()
from keras.optimizers import Adam
adam = Adam(lr=0.005)
model.compile(loss=['mean_squared_error','categorical_crossentropy'], optimizer=adam,loss_weights=[.005, 10.],metrics=['accuracy'])

# This stores the history of the training to e.g. allow to plot the learning curve
from keras.callbacks import History # , TensorBoard
# loss per epoch
history = History()


dc=DataCollection()
dc.readFromFile('/Users/jkiesele/Cernbox/batchtest/file.txt')
dtrain=dc.splitToTrainAndTest(0.8)[0]
dtest=dc.splitToTrainAndTest(0.8)[1]

#
# things to add:
# maxqueues= system.memory / dc.getMemoryPerUnit will read one file and make an estimate


model.fit_generator(dtrain.generator(TrainData_veryDeepJet()) ,
        samples_per_epoch=1, nb_epoch=3,max_q_size=1,callbacks=[history],
        validation_data=dtest.generator(TrainData_veryDeepJet()),
        nb_val_samples=1)


# the actual training
#model.fit([x_local, x_global], [reg_truth,class_truth] ,validation_split=0.99, nb_epoch=20, batch_size=1000, callbacks=[history], sample_weight=[weights,weights])


print(history.history.keys())
# summarize history for loss for trainin and test sample
plt.plot(history.history['loss'])
#print(history.history['val_loss'],history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(outputDir+'learningcurve.pdf') 
plt.close()

plt.plot(history.history['dense_7_acc'])
plt.plot(history.history['val_dense_7_acc'])
plt.title('model d7 accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(outputDir+'d7_accuracy.pdf') 
plt.close()

plt.plot(history.history['dense_8_acc'])
plt.plot(history.history['val_dense_8_acc'])
plt.title('model d8 accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(outputDir+'d8_accuracy.pdf') 
#plt.show()

#from keras.models import load_model
model.save(outputDir+"KERAS_model.h5")
