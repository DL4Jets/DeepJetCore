from DeepJetCore.training.training_base import training_base
from tools import model_deepFlavourReference_test



train=training_base(testrun=False)
newtraining= not train.modelSet()


if newtraining:
    train.setModel(model_deepFlavourReference_test)
    
    #train.keras_model=fixLayersContaining(train.keras_model, 'regression', invert=False)
    
    train.compileModel(learningrate=0.001,
                       loss=['categorical_crossentropy'],
                       metrics=['accuracy'])


    train.train_data.maxFilesOpen=5
    print(train.keras_model.summary())
    model,history = train.trainModel(nepochs=3, 
                                     batchsize=10000, 
                                     stop_patience=300, 
                                     lr_factor=0.5, 
                                     lr_patience=3, 
                                     lr_epsilon=0.0001, 
                                     lr_cooldown=6, 
                                     lr_minimum=0.0001, 
                                     maxqsize=150,
                                     plot_batch_loss = True)


