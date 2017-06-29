

from training_base import training_base
from Losses import loss_NLL

#also does all the parsing
train=training_base(testrun=True)


if not train.modelSet():
    from models import convolutional_model_broad_map_reg
    
    train.setModel(convolutional_model_broad_map_reg,dropoutRate=0.1)
    
    train.compileModel(learningrate=0.005,
                       loss=['categorical_crossentropy',loss_NLL],
                       metrics=['accuracy'],
                       loss_weights=[1., 0.00001])


model,history = train.trainModel(nepochs=5, 
                                 batchsize=250, 
                                 stop_patience=300, 
                                 lr_factor=0.5, 
                                 lr_patience=10, 
                                 lr_epsilon=0.0001, 
                                 lr_cooldown=2, 
                                 lr_minimum=0.0001, 
                                 maxqsize=100)