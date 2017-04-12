'''
Created on 6 Mar 2017

@author: jkiesele
'''
from keras.callbacks import Callback



class learningRateDecrease(Callback):
    ''' 
    kicks in after ep_start epochs
    decreases the learning rate every n_epoch 
    with exponential decrease of lr_decrease
    until value lr_thresh is reached (default 0)
    '''  
    def __init__(self, n_epoch, decay , startlr, ep_start=1,lr_thresh=0):
        super(learningRateDecrease, self).__init__()
        self.n_epoch=n_epoch
        self.decay=decay
        self.ep_start=ep_start
        self.lr_thresh=lr_thresh
        self.startlr=startlr
        self.__learnrate=startlr
        self.__mode=0

    def setExponentialMode(self):
        self.__mode=1
        
        
    def setStepMode(self):
        self.__mode=0

    def reducelearnrate(self, epoch):
        if epoch==0 :
            self.__learnrate=self.startlr
            
        if epoch > self.ep_start and epoch%self.n_epoch == 0 and self.lr_thresh < self.__learnrate:
            if self.mode == 1:
                self.__learnrate= self.decay*self.__learnrate
            elif self.mode == 0:
                self.__learnrate-=self.decay
            return self.__learnrate
        else:
            return self.__learnrate


