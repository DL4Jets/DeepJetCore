

from training_base import training_base
from Losses import loss_NLL

#also does all the parsing
train=training_base(testrun=True)


if not train.modelSet():
    from models import dense_model
    
    train.setModel(dense_model,dropoutRate=0.1)
    
    train.compileModel(learningrate=0.003,
                       loss='categorical_crossentropy',
                       metrics='accuracy')


model,history = train.trainModel(nepochs=50, 
                                 batchsize=5000, 
                                 stop_patience=300, 
                                 lr_factor=0.5, 
                                 lr_patience=10, 
                                 lr_epsilon=0.0001, 
                                 lr_cooldown=2, 
                                 lr_minimum=0.0001, 
                                 maxqsize=100)