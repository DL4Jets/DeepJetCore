'''
Created on 7 Apr 2017

@author: jkiesele
'''
from __future__ import print_function

from .ReduceLROnPlateau import ReduceLROnPlateau
from ..evaluation import plotLoss

from keras.callbacks import Callback, EarlyStopping,History,ModelCheckpoint #, ReduceLROnPlateau # , TensorBoard
# loss per epoch
from time import time
from pdb import set_trace
import json
from keras import backend as K
import matplotlib
import os
matplotlib.use('Agg') 


class plot_loss_or_metric(Callback):
    def __init__(self,outputDir,metrics):
        self.metrics=metrics
        self.outputDir=outputDir
        
    def on_epoch_end(self,epoch, epoch_logs={}):
        lossfile=os.path.join( self.outputDir, 'full_info.log')
        allinfo_history=None
        with open(lossfile, 'r') as infile:
            allinfo_history=json.load(infile)
            
        nepochs=len(allinfo_history)
        allnumbers=[[] for i in range(len(self.metrics))]
        epochs=[]
        for i in range(nepochs):
            epochs.append(i)
            for j in range(len(self.metrics)):
                allnumbers[j].append(allinfo_history[i][self.metrics[j]])
        
        import matplotlib.pyplot as plt
        for j in range(len(self.metrics)):
            f = plt.figure()
            plt.plot(epochs,allnumbers[j],'r',label=self.metrics[j])
            plt.ylabel(self.metrics[j])
            plt.xlabel('epoch')
            #plt.legend()
            f.savefig(self.outputDir+'/'+self.metrics[j]+'.pdf')
            plt.close()
    

class newline_callbacks_begin(Callback):
    
    def __init__(self,outputDir,plotLoss=False):
        self.outputDir=outputDir
        self.loss=[]
        self.val_loss=[]
        self.full_logs=[]
        self.plotLoss=plotLoss
        
    def on_epoch_end(self,epoch, epoch_logs={}):
        import os
        lossfile=os.path.join( self.outputDir, 'losses.log')
        print('\n***callbacks***\nsaving losses to '+lossfile)
        self.loss.append(epoch_logs.get('loss'))
        self.val_loss.append(epoch_logs.get('val_loss'))
        f = open(lossfile, 'a')
        f.write(str(epoch_logs.get('loss')))
        f.write(" ")
        f.write(str(epoch_logs.get('val_loss')))
        f.write("\n")
        f.close()    
        learnfile=os.path.join( self.outputDir, 'learn.log')
        with open(learnfile, 'a') as f:
            f.write(str(float(K.get_value(self.model.optimizer.lr)))+'\n')
        
        lossfile=os.path.join( self.outputDir, 'full_info.log')
        if os.path.isfile(lossfile):
            with open(lossfile, 'r') as infile:
                self.full_logs=json.load(infile)
            
        normed = {}
        for vv in epoch_logs:
            normed[vv] = float(epoch_logs[vv])
        self.full_logs.append(normed)
        
        with open(lossfile, 'w') as out:
            out.write(json.dumps(self.full_logs))
            
        if self.plotLoss:
            plotLoss(self.outputDir+'/losses.log',self.outputDir+'/losses.pdf',[])
        
class newline_callbacks_end(Callback):
    def on_epoch_end(self,epoch, epoch_logs={}):
        print('\n***callbacks end***\n')
        
        
class Losstimer(Callback):
    def __init__(self, every = 50):
        self.points = []
        self.every = every
        self.counter=0

    def on_train_begin(self, logs):
        self.start = time()

    def on_batch_end(self, batch, logs):
        if (self.counter != self.every): 
            self.counter+=1
            return
        self.counter = 0
        elapsed = time() - self.start
        cop = {}
        for i, j in logs.iteritems():
            cop[i] = float(j)
        cop['elapsed'] = elapsed
        self.points.append(cop)
        
        
class checkTokens_callback(Callback):
    
    def __init__(self,cutofftime_hours=48):
        self.cutofftime_hours=cutofftime_hours
        
    def on_epoch_begin(self, epoch, logs=None):
        from tokenTools import checkTokens
        checkTokens(self.cutofftime_hours)
        
class saveCheckPointDeepJet(Callback):
    '''
    this seems obvious, however for some reason the keras model checkpoint fails
    to save the optimizer state, needed for resuming a training. Therefore this explicit
    implementation.
    '''
    
    def __init__(self,outputDir,model):
        self.outputDir=outputDir
        self.djmodel=model
    def on_epoch_end(self,epoch, epoch_logs={}):
        self.djmodel.save(self.outputDir+"/KERAS_check_model_last.h5")
        
        
class DeepJet_callbacks(object):
    def __init__(self,
                 model,
                 stop_patience=-1,
                 lr_factor=0.5,
                 lr_patience=-1,
                 lr_epsilon=0.001,
                 lr_cooldown=4,
                 lr_minimum=1e-5,
                 outputDir='',
                 minTokenLifetime=5,
                 checkperiod=10,
                 checkperiodoffset=0,
                 plotLossEachEpoch=True, 
                 additional_plots=None):
        

        
        self.nl_begin=newline_callbacks_begin(outputDir,plotLossEachEpoch)
        self.nl_end=newline_callbacks_end()
        
        self.callbacks=[self.nl_begin]
        
        if minTokenLifetime>0:
            self.tokencheck=checkTokens_callback(minTokenLifetime)
            self.callbacks.append(self.tokencheck)
        
        
        if lr_patience>0:
            self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=lr_factor, patience=lr_patience, 
                                    mode='min', verbose=1, epsilon=lr_epsilon,
                                     cooldown=lr_cooldown, min_lr=lr_minimum)
            self.callbacks.append(self.reduce_lr)


        self.modelbestcheck=ModelCheckpoint(outputDir+"/KERAS_check_best_model.h5", 
                                        monitor='val_loss', verbose=1, 
                                        save_best_only=True, save_weights_only=False)
        self.callbacks.append(self.modelbestcheck)
        
        if checkperiod>0:
            self.modelcheckperiod=ModelCheckpoint(outputDir+"/KERAS_check_model_block_"+str(checkperiodoffset)+"_epoch_{epoch:02d}.h5", 
                                                  verbose=1,period=checkperiod, save_weights_only=False)
            self.callbacks.append(self.modelcheckperiod)
        
        self.modelcheck=saveCheckPointDeepJet(outputDir,model)
        self.callbacks.append(self.modelcheck)
        
        if stop_patience>0:
            self.stopping = EarlyStopping(monitor='val_loss', 
                                          patience=stop_patience, 
                                          verbose=1, mode='min')
            self.callbacks.append(self.stopping)
        
        if additional_plots:
            self.additionalplots = plot_loss_or_metric(outputDir,additional_plots)
            self.callbacks.append(self.additionalplots)
            
        self.history=History()
        self.timer = Losstimer()
        
  
        self.callbacks.extend([ self.nl_end, self.history,self.timer])
        
        
        
from DeepJetCore.TrainData import TrainData

class PredictCallback(Callback):
    
    def __init__(self, 
                 samplefile='',
                 function_to_apply=None, #needs to be function(counter,[model_input], [predict_output], [truth])
                 after_n_batches=50,
                 on_epoch_end=False,
                 use_event=0
                 ):
        super(PredictCallback, self).__init__()
        self.samplefile=samplefile
        self.function_to_apply=function_to_apply
        self.counter=0
        
        self.after_n_batches=after_n_batches
        self.run_on_epoch_end=on_epoch_end
        
        if self.run_on_epoch_end and self.after_n_batches>=0:
            print('PredictCallback: can only be used on epoch end OR after n batches, falling back to epoch end')
            self.after_n_batches=0
        
        self.td=TrainData()
        self.td.readIn(samplefile)
        self.td.skim(event=use_event)
        
    def on_train_begin(self, logs=None):
        pass
    
    def predict_and_call(self,counter):
        
        predicted = self.model.predict(self.td.x)
        if not isinstance(predicted, list):
            predicted=[predicted]
        
        self.function_to_apply(counter,self.td.x,predicted,self.td.y)
    
    def on_epoch_end(self, epoch, logs=None):
        self.counter=0
        if not self.run_on_epoch_end: return
        self.predict_and_call(epoch)
        
    def on_batch_end(self, batch, logs=None):
        if self.after_n_batches<=0: return
        self.counter+=1
        if self.counter>self.after_n_batches: 
            self.counter=0
            self.predict_and_call(batch)
        
        
           
        
        
        
        
        
        

