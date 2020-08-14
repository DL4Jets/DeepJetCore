'''
Created on 7 Apr 2017

@author: jkiesele
'''

import matplotlib
matplotlib.use('Agg') 


from .ReduceLROnPlateau import ReduceLROnPlateau
from ..evaluation import plotLoss
from ..evaluation import plotBatchLoss

import matplotlib.pyplot as plt
import numpy as np

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
        
    def on_epoch_end(self,epoch, logs={}):
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
        
    def on_epoch_end(self,epoch, logs={}):
        if len(logs)<1:
            return
        import os
        lossfile=os.path.join( self.outputDir, 'losses.log')
        print('\n***callbacks***\nsaving losses to '+lossfile)
        
        # problem with new keras version calling callbacks even after exceptions
        if logs.get('loss') is None:
            return 
        if logs.get('val_loss') is None:
            return
        
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        f = open(lossfile, 'a')
        f.write(str(logs.get('loss')))
        f.write(" ")
        f.write(str(logs.get('val_loss')))
        f.write("\n")
        f.close()    
        learnfile=os.path.join( self.outputDir, 'learn.log')
        try:
            with open(learnfile, 'a') as f:
                f.write(str(float(K.get_value(self.model.optimizer.lr)))+'\n')
            
            lossfile=os.path.join( self.outputDir, 'full_info.log')
            if os.path.isfile(lossfile):
                with open(lossfile, 'r') as infile:
                    self.full_logs=json.load(infile)
                
            normed = {}
            for vv in logs:
                normed[vv] = float(logs[vv])
            self.full_logs.append(normed)
            
            with open(lossfile, 'w') as out:
                out.write(json.dumps(self.full_logs))
        except:
            pass
                
        if self.plotLoss:
            try:
                plotLoss(self.outputDir+'/losses.log',self.outputDir+'/losses.pdf',[])
            except:
                pass

class batch_callback_begin(Callback):

    def __init__(self,outputDir,plotLoss=False,plot_frequency=-1,batch_frequency=1):
        self.outputDir=outputDir
        self.loss=[]
        self.val_loss=[]
        self.full_logs=[]
        self.plotLoss=plotLoss
        self.plot_frequency=plot_frequency
        self.plotcounter=0
        self.batch_frequency=batch_frequency
        self.batchcounter=0


    
        
    def read(self):
        
        import os
        if not os.path.isfile(self.outputDir+'/batch_losses.log') :
            return
        blossfile=os.path.join( self.outputDir, 'batch_losses.log')
        f = open(blossfile, 'r')
        self.loss = []
        for line in f:
            if len(line)<1: continue
            tl=float(line.split(' ')[0])
            self.loss.append(tl)
        
        f.close() 
            
    def on_batch_end(self,batch,logs={}):
        if len(logs)<1:
            return
        if logs.get('loss') is None:
            return 
        self.batchcounter += 1
        
        if not self.batch_frequency == self.batchcounter:
            return
        self.batchcounter=0
        
        self.loss.append(logs.get('loss'))
        
        if self.plot_frequency<0:
            return 
        self.plotcounter+=1
        if self.plot_frequency == self.plotcounter:
            self.plot()
            self.plotcounter = 0
         
        
    def _plot(self):
        if len(self.loss) < 2:
            return 
        batches = [self.batch_frequency*i for i in range(len(self.loss))]
        plt.close()
        plt.plot(batches,self.loss,'r-',label='loss')
        
        def smooth(y, box_pts):
            box = np.ones(box_pts)/box_pts
            y_smooth = np.convolve(y, box, mode='same')
            return y_smooth
        
        if len(batches) > 50:
            smoothed = smooth(self.loss,50)
            #remove where the simple smoothing doesn't give reasonable results
            plt.plot(batches[25:-25],smoothed[25:-25],'g-',label='smoothed')
            plt.legend()
        
        plt.xlabel("# batches")
        plt.ylabel("training loss")
        plt.yscale("log")
        plt.savefig(self.outputDir+'/batch_losses.pdf')
        plt.close()
        
     
    def plot(self):
        self._plot()
        
    def save(self):
        
        import os
        blossfile=os.path.join( self.outputDir, 'batch_losses.log')
        f = open(blossfile, 'w')
        for i in range(len(self.loss)):
            f.write(str(self.loss[i]))
            f.write("\n")
        self.loss=[]
        self.val_loss=[]
        f.close()      
        
        
    def on_epoch_end(self,epoch,logs={}):
        self.plot()
        self.save()
        
    def on_epoch_begin(self, epoch, logs=None):
        self.read()
        if len(self.loss):
            self.plot()
        
class newline_callbacks_end(Callback):
    def on_epoch_end(self,epoch, logs={}):
        print('\n***callbacks end***\n')
        
        
class Losstimer(Callback):
    def __init__(self, every = 50):
        self.points = []
        self.every = every
        self.counter=0

    def on_train_begin(self, logs):
        self.start = time()

    def on_batch_end(self, batch, logs={}):
        if (self.counter != self.every): 
            self.counter+=1
            return
        self.counter = 0
        elapsed = time() - self.start
        cop = {}
        for i, j in logs.items():
            cop[i] = float(j)
        cop['elapsed'] = elapsed
        self.points.append(cop)
        
        
class checkTokens_callback(Callback):
    
    def __init__(self,cutofftime_hours=48):
        self.cutofftime_hours=cutofftime_hours
        
    def on_epoch_begin(self, epoch, logs=None):
        from .tokenTools import checkTokens
        checkTokens(self.cutofftime_hours)
        
class saveCheckPointDeepJet(Callback):
    '''
    Slight extension of the normal checkpoint to multiple checkpoints per epoch
    '''
    
    def __init__(self,outputFile,model,check_n_batches=-1,nrotate=3):
        self.outputFile=outputFile
        self.djmodel=model
        self.counter=0
        self.rotate_idx=0
        self.rotations=[str(i) for i in range(nrotate)]
        self.check_n_batches=check_n_batches
        
    def on_batch_end(self,batch,logs={}):
        if self.check_n_batches < 1:
            return
        if self.counter < self.check_n_batches:
            self.counter+=1
            return
        self.djmodel.save(self.outputFile[:-3]+'_rot_'+self.rotations[self.rotate_idx]+'.h5')
        self.djmodel.save(self.outputFile)
        self.counter=0
        self.rotate_idx += 1
        if self.rotate_idx >= len(self.rotations):
            self.rotate_idx=0
        
    def on_epoch_end(self,epoch, logs={}):
        if len(logs)<1:
            return
        if logs.get('loss') is None:
            return 
        if logs.get('val_loss') is None:
            return
        self.djmodel.save(self.outputFile)
        
        
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
                 backup_after_batches=-1,
                 checkperiodoffset=0,
                 plotLossEachEpoch=True, 
                 additional_plots=None,
                 batch_loss = False):
        

        
        self.nl_begin=newline_callbacks_begin(outputDir,plotLossEachEpoch)
        self.nl_end=newline_callbacks_end()
        
        self.callbacks=[self.nl_begin]
        
        if batch_loss:
            self.batch_callback=batch_callback_begin(outputDir,plotLossEachEpoch)
            self.callbacks.append(self.batch_callback)

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
        
        self.modelcheck=saveCheckPointDeepJet(outputDir+"/KERAS_check_model_last.h5",model,backup_after_batches)
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
from DeepJetCore.dataPipeline import TrainDataGenerator

class PredictCallback(Callback):
    
    def __init__(self, 
                 samplefile,
                 function_to_apply=None, #needs to be function(counter,[model_input], [predict_output], [truth])
                 after_n_batches=50,
                 batchsize=10,
                 on_epoch_end=False,
                 use_event=0,
                 decay_function=None,
                 offset=0
                 ):
        super(PredictCallback, self).__init__()
        self.samplefile=samplefile
        self.function_to_apply=function_to_apply
        self.counter=0
        self.call_counter=offset
        self.decay_function=decay_function
        
        self.after_n_batches=after_n_batches
        self.run_on_epoch_end=on_epoch_end
        
        if self.run_on_epoch_end and self.after_n_batches>=0:
            print('PredictCallback: can only be used on epoch end OR after n batches, falling back to epoch end')
            self.after_n_batches=0
        
        td=TrainData()
        td.readFromFile(samplefile)
        if use_event>=0:
            td.skim(use_event)
            
        self.batchsize = 1    
        self.td = td
        self.gen = TrainDataGenerator()
        self.gen.setBatchSize(batchsize)
        self.gen.setSkipTooLargeBatches(False)

    
    def reset(self):
        self.call_counter=0
    
    def predict_and_call(self,counter):
        
        self.gen.setBuffer(self.td)
        
        predicted = self.model.predict_generator(self.gen.feedNumpyData(),
                                            steps=self.gen.getNBatches(),
                                            max_queue_size=1,
                                            use_multiprocessing=False,
                                            verbose=2)
        
        if not isinstance(predicted, list):
            predicted=[predicted]
        
        self.function_to_apply(self.call_counter,self.td.copyFeatureListToNumpy(),
                               predicted,self.td.copyTruthListToNumpy())
        self.call_counter+=1
    
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
            if self.decay_function is not None:
                self.after_n_batches=self.decay_function(self.call_counter)
        
        
           
        
        
        
        
        
        

