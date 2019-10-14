from DeepJetCore.training.training_base import training_base
from DeepJetCore.TrainData import TrainData
from DeepJetCore.TrainData import fileTimeOut as tdfto
import numpy as np
from DeepJetCore.evaluation import plotLoss



from keras.callbacks import Callback
from keras.layers import Dense
from keras.models import Model




def model_for_test(Inputs,nclasses,nregclasses):

    globalvars = (Inputs[0])
    
    x = Dense(50,activation='relu',kernel_initializer='lecun_uniform')(globalvars)
    
    flavour_pred=Dense(nclasses, activation='softmax',kernel_initializer='lecun_uniform',name='ID_pred')(x)
    
    predictions = [flavour_pred]
    model = Model(inputs=Inputs, outputs=predictions)
    return model







train=training_base(testrun=False)
newtraining= not train.modelSet()

def testequal(xout,yout):
    
    print(np.all(xout[0]==yout[0][:,0:1]))

#train.train_data.test_output_function=testequal



if newtraining:
    train.setModel(model_for_test)
    
    #train.keras_model=fixLayersContaining(train.keras_model, 'regression', invert=False)
    
    train.compileModel(learningrate=0.001,
                       loss=['categorical_crossentropy'],
                       metrics=['accuracy'])


    train.train_data.maxFilesOpen=5
print(train.keras_model.summary())
model,history = train.trainModel(nepochs=10, 
                                 verbose=1,
                                     batchsize=10000,
                                     
                                     load_in_mem = False,
                                     plot_batch_loss = True)


