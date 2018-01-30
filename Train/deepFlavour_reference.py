

from training_base import training_base
from Losses import loss_NLL, loss_meansquared
from modelTools import fixLayersContaining,printLayerInfosAndWeights

#also does all the parsing
train=training_base(testrun=False)

newtraining= not train.modelSet()
#for recovering a training
if newtraining:
    from models import model_deepFlavourReference
    
    train.setModel(model_deepFlavourReference,dropoutRate=0.1,momentum=0.3)
    
    #train.keras_model=fixLayersContaining(train.keras_model, 'regression', invert=False)
    
    train.compileModel(learningrate=0.001,
                       loss=['categorical_crossentropy',loss_meansquared],
                       metrics=['accuracy'],
                       loss_weights=[1., 0.000000000001])


    train.train_data.maxFilesOpen=5
    
    print(train.keras_model.summary())
    model,history = train.trainModel(nepochs=1, 
                                     batchsize=10000, 
                                     stop_patience=300, 
                                     lr_factor=0.5, 
                                     lr_patience=3, 
                                     lr_epsilon=0.0001, 
                                     lr_cooldown=6, 
                                     lr_minimum=0.0001, 
                                     maxqsize=5)
    
    
    print('fixing input norms...')
    train.keras_model=fixLayersContaining(train.keras_model, 'input_batchnorm')
    train.compileModel(learningrate=0.0003,
                           loss=['categorical_crossentropy',loss_meansquared],
                           metrics=['accuracy'],
                           loss_weights=[1., 0.000000000001])
    
print(train.keras_model.summary())
#printLayerInfosAndWeights(train.keras_model)

model,history = train.trainModel(nepochs=63, #sweet spot from looking at the testing plots 
                                 batchsize=10000, 
                                 stop_patience=300, 
                                 lr_factor=0.8, 
                                 lr_patience=-3, 
                                 lr_epsilon=0.0001, 
                                 lr_cooldown=8, 
                                 lr_minimum=0.00001, 
                                 maxqsize=5,verbose=1)
