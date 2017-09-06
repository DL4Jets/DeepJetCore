

from training_base import training_base
from Losses import loss_NLL
from modelTools import fixLayersContaining,printLayerInfosAndWeights

#also does all the parsing
train=training_base(testrun=False)

newtraining= not train.modelSet()
#for recovering a training
if newtraining:
    from models import model_deepFlavourNoNeutralReference
    
    train.setModel(model_deepFlavourNoNeutralReference,dropoutRate=0.1)
    
    #train.keras_model=fixLayersContaining(train.keras_model, 'regression', invert=False)
    
    train.compileModel(learningrate=0.001,
                       loss=['categorical_crossentropy',loss_NLL],
                       metrics=['accuracy'],
                       loss_weights=[1., 0.000000000001])


print(train.keras_model.summary())
model,history = train.trainModel(nepochs=1, 
                                 batchsize=10000, 
                                 stop_patience=300, 
                                 lr_factor=0.5, 
                                 lr_patience=3, 
                                 lr_epsilon=0.0001, 
                                 lr_cooldown=6, 
                                 lr_minimum=0.0001, 
                                 maxqsize=100)


print('fixing input norms...')
train.keras_model=fixLayersContaining(train.keras_model, 'input_batchnorm')
train.compileModel(learningrate=0.001,
                       loss=['categorical_crossentropy',loss_NLL],
                       metrics=['accuracy'],
                       loss_weights=[1., 0.000000000001])


print(train.keras_model.summary())
#printLayerInfosAndWeights(train.keras_model)

model,history = train.trainModel(nepochs=60, 
                                 batchsize=10000, 
                                 stop_patience=300, 
                                 lr_factor=0.5, 
                                 lr_patience=3, 
                                 lr_epsilon=0.0001, 
                                 lr_cooldown=6, 
                                 lr_minimum=0.0001, 
                                 maxqsize=100)
