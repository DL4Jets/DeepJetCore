

from training_base import training_base


train=training_base()

from DeepJet_models import Test_model

train.setModel(Test_model,dropoutRate=0.1)

train.compileModel(learningrate=0.005,
                   loss='categorical_crossentropy',metrics=['accuracy'])

train.trainModel(nepochs=5, 
                 batchsize=250, 
                 stop_patience=300, 
                 lr_factor=0.5, 
                 lr_patience=10, 
                 lr_epsilon=0.0001, 
                 lr_cooldown=2, 
                 lr_minimum=0.0001, 
                 maxqsize=10)