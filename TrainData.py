'''
Created on 20 Feb 2017

@author: jkiesele

New (post equals 2.0) version
'''

from __future__ import print_function

import os
import numpy as np
import logging

from DeepJetCore.compiled import c_trainDataInterface as ctd

def fileTimeOut(fileName, timeOut):
    '''
    simple wait function in case the file system has a glitch.
    waits until the dir, the file should be stored in/read from, is accessible
    again, or the the timeout
    '''
    filepath=os.path.dirname(fileName)
    if len(filepath) < 1:
        filepath = '.'
    if os.path.isdir(filepath):
        return

    counter=0
    print('file I/O problems... waiting for filesystem to become available for '+fileName)
    while not os.path.isdir(filepath):
        if counter > timeOut:
            print('...file could not be opened within '+str(timeOut)+ ' seconds')
        counter+=1
        time.sleep(1)



class TrainData(object):
    '''
    Base class for batch-wise training of the DNN
    '''
    
    def __init__(self):
        '''
        Constructor
        
        '''
        self.clear()
        
        
    def __del__(self):
        self.clear()
        

    def clear(self):
        
        if hasattr(self, 'x'):
            del self.x
            del self.y
            del self.w
        if hasattr(self, 'w_list'):
            del self.w_list
            del self.x_list
            del self.y_list
            
        self.x=[]
        self.y=[]
        self.w=[]
        
        self.xshapes=[]
        self.yshapes=[]
        self.wshapes=[]
        
        self.nsamples=None
        
    def skim(self, event=0):
        xs=[]
        ys=[]
        ws=[]
        
        for x in self.x:
            xs.append(x[event:event+1,...])
        for y in self.y:
            ys.append(y[event:event+1,...])
        for w in self.w:
            ws.append(w[event:event+1,...])
        self.clear()
        self.nsamples=1
        self.x=xs
        self.y=ys
        self.w=ws 
        
    # to be defined by user implementation
    def definePredictionToRoot(self, prediction):
        pass 
    
    def _getShapes(self, arrlist):
        
        outl=[]
        for x in arrlist:
            outl.append(x.shape)
        shapes=[]
        for s in outl:
            _sl=[]
            for i in range(len(s)):
                if i:
                    _sl.append(s[i])
            s=(_sl)
            if len(s)==0:
                s.append(1)
            shapes.append(s)
    
    def getFeatureShapes(self):
        if not len(self.xshapes):
            self.xshapes = _getShapes(x)
        return self.xshapes
    
    def getInputShapes(self):
        print('TrainData:getInputShapes: Deprecated, use getFeatureShapes instead')
        return getFeatureShapes()
        
    def getTruthShapes(self):
        if not len(self.yshapes):
            self.yshapes = _getShapes(y)
        return self.yshapes
    
    def writeToFile(self,filename):
        ctd.writeToFile(self.x,self.y,self.w,filename)
       
    def readFromFile(self,fileprefix,shapesOnly=False):
        self.x=[]
        self.y=[]
        self.w=[]
        if shapesOnly:
            
            ### needs rework
            shapes = ctd.readShapesFromFile(fileprefix)
            self.xshapes = shapes[0][1:]
            self.yshapes = []
            if len(shapes[1]):
                self.yshapes = shapes[1][1:]
            self.wshapes = []
            if len(shapes[2]):
                self.wshapes = shapes[2][1:]
                
            ###
            return
        l = ctd.readFromFile(fileprefix)
        self.x = l[0]
        self.y = l[1]
        self.w = l[2]
        self.nsamples=len(x[0])
        
    def readIn(self,fileprefix,shapesOnly=False):
        print('TrainData:readIn deprecated, use readFromFile')
        self.readFromFile(fileprefix,shapesOnly)
        
        
    def convertFromSourceFile(self, filename, weighterobjects):
        pass
    
    def createWeighterObjects(self, allsourcefiles):
        '''
        Will be called on the full list of source files once.
        Can be used to create weighter objects or similar that can
        then be applied to each individual conversion
        '''
        return ()
    

