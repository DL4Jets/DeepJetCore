

from training_base import training_base
from Losses import loss_NLL

#also does all the parsing
train=training_base(testrun=False)


if not train.modelSet():
    from DeepJet_models_ResNet import resnet_model
    
    train.setModel(resnet_model)
    
    train.compileModel(learningrate=0.0004,
                       loss=['categorical_crossentropy',loss_NLL],
                       metrics=['accuracy'],
                       loss_weights=[1., 0.000000000000001])


model,history = train.trainModel(nepochs=50, 
                                 batchsize=10000, 
                                 stop_patience=300, 
                                 lr_factor=0.5, 
                                 lr_patience=10, 
                                 lr_epsilon=0.0001, 
                                 lr_cooldown=2, 
                                 lr_minimum=0.0001, 
                                 maxqsize=100)