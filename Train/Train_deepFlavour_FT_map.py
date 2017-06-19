

from training_base import training_base

#also dows all the parsing
train=training_base()

from models import convolutional_model_broad_map

train.setModel(convolutional_model_broad_map,dropoutRate=0.1)

train.compileModel(learningrate=0.0005,
                   loss=['categorical_crossentropy'],
                   metrics=['accuracy'])


model,history = train.trainModel(nepochs=100, 
                                 batchsize=10000, 
                                 stop_patience=300, 
                                 lr_factor=0.5, 
                                 lr_patience=10, 
                                 lr_epsilon=0.003, 
                                 lr_cooldown=6, 
                                 lr_minimum=0.000001, 
                                 maxqsize=100)



######
