'''
Created on 7 Apr 2017

@author: jkiesele
'''
from __future__ import print_function

from ReduceLROnPlateau import ReduceLROnPlateau
from keras.callbacks import Callback, EarlyStopping,History,ModelCheckpoint #, ReduceLROnPlateau # , TensorBoard
# loss per epoch


class newline_callbacks_begin(Callback):
    def on_epoch_end(self,epoch, epoch_logs={}):
        print('\n***callbacks***\n')
        
class newline_callbacks_end(Callback):
    def on_epoch_end(self,epoch, epoch_logs={}):
        print('\n***callbacks end***\n')
        
        
        
class DeepJet_callbacks(object):
    def __init__(self,
                 stop_patience=10,
                 lr_factor=0.5,
                 lr_patience=1,
                 lr_epsilon=0.001,
                 lr_cooldown=4,
                 lr_minimum=1e-5,
                 outputDir=''):
        
        
        self.nl_begin=newline_callbacks_begin()
        self.nl_end=newline_callbacks_end()
        
        self.stopping = EarlyStopping(monitor='val_loss', 
                                      patience=stop_patience, 
                                      verbose=1, mode='min')
        
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=lr_factor, patience=lr_patience, 
                                mode='min', verbose=1, epsilon=lr_epsilon,
                                 cooldown=lr_cooldown, min_lr=lr_minimum)

        self.modelbestcheck=ModelCheckpoint(outputDir+"/KERAS_check_best_model.h5", 
                                        monitor='val_loss', verbose=1, 
                                        save_best_only=True)
        
        self.modelcheck=ModelCheckpoint(outputDir+"/KERAS_check_last_model.h5", verbose=1)
  
        self.history=History()
  
        self.callbacks=[self.nl_begin,self.modelbestcheck,self.modelcheck,self.reduce_lr, self.stopping,self.nl_end,self.history]
