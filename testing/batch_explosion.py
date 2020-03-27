from DeepJetCore.training.training_base import training_base
from DeepJetCore.TrainData import TrainData
from DeepJetCore.TrainData import fileTimeOut as tdfto
import numpy as np
from DeepJetCore.evaluation import plotLoss



from keras.callbacks import Callback
from keras.layers import Dense
from keras.models import Model
from DeepJetCore.DJCLayers import  Print



def model_for_test(Inputs):

    x = Inputs[0]
    x = Dense(50,activation='relu')(x)
    x = Dense(2, activation='softmax')(x)
    
    predictions = [x]
    model = Model(inputs=Inputs, outputs=predictions)
    return model



train=training_base(testrun=False)

from DeepJetCore.DJCLosses import dummy_loss


train.setModel(model_for_test)


train.compileModel(learningrate=0.001,
                   loss='categorical_crossentropy',metrics=['accuracy'])
                   #metrics=['accuracy'])


print(train.keras_model.summary())
model,history = train.trainModel(nepochs=10, 
                                 verbose=2,
                                     batchsize=1000,
                                     load_in_mem = False,
                                     plot_batch_loss = False)


