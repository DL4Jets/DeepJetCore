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
        
        self.sourcefile=""
        
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
        self.x=xs
        self.y=ys
        self.w=ws 
        for s in self.xshapes:
            s[0]=1
        for s in self.yshapes:
            s[0]=1
        for s in self.wshapes:
            s[0]=1
        
    def getNTotal(self):
        if len(self.xshapes) == 0 or len(self.xshapes[0]) == 0:
            return 0
        return self.xshapes[0][0]
    
    def getKerasFeatureShapes(self):
        return [a[1:] for a in self.xshapes]
    
    def getInputShapes(self):
        print('TrainData:getInputShapes: Deprecated, use getKerasFeatureShapes instead')
        return getKerasFeatureShapes()
        
    def getKerasTruthShapes(self):
        return [a[1:] for a in self.yshapes]
    
    def writeToFile(self,filename):
        ctd.writeToFile(self.x,self.y,self.w,filename)
       
    def readFromFile(self,infile,shapesOnly=False):
        '''
        For debugging or getting shapes.
        Don't use this function for a generator, use the C++ Generator instead!
        '''
        self.clear()
        self.sourcefile=infile

        print('infile',infile)
        shapes = ctd.readShapesFromFile(infile)
        self.xshapes = shapes[0]
        self.yshapes = shapes[1]
        self.wshapes = shapes[2]
            
        ###
        if shapesOnly:
            return
        l = ctd.readFromFile(fileprefix)
        self.x = l[0]
        self.y = l[1]
        self.w = l[2]
        self.nsamples=len(x[0])
        
    def readIn(self,fileprefix,shapesOnly=False):
        print('TrainData:readIn deprecated, use readFromFile')
        self.readFromFile(fileprefix,shapesOnly)
        
        
        
    ################# functions to be defined by the user    
        
    def createWeighterObjects(self, allsourcefiles):
        '''
        Will be called on the full list of source files once.
        Can be used to create weighter objects or similar that can
        then be applied to each individual conversion.
        Should return a dictionary
        '''
        return {}
    
    def convertFromSourceFile(self, filename, weighterobjects):
        pass
    
    def writeOutPrediction(self, predicted, features, truth, weights, outfilename):
        pass

