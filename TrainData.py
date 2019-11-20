'''
Created on 20 Feb 2017

@author: jkiesele

New (post equals 2.1) version
'''

from __future__ import print_function

import os
import numpy as np
import logging

from DeepJetCore.compiled.c_trainData import trainData
from DeepJetCore.compiled.c_simpleArray import simpleArray

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

#inherit from cpp class, just slim wrapper

class TrainData(trainData):
    '''
    Base class for batch-wise training of the DNN
    '''
    def __init__(self):
        trainData.__init__(self)
        
    
    def getInputShapes(self):
        print('TrainData:getInputShapes: Deprecated, use getKerasFeatureShapes instead')
        return self.getKerasFeatureShapes()
        
    
    def readIn(self,fileprefix,shapesOnly=False):
        print('TrainData:readIn deprecated, use readFromFile')
        self.readFromFile(fileprefix,shapesOnly)
    
    
    def _maybeConvertToSimpleArray(self,a):
        if str(type(a)) == "<class 'DeepJetCore.compiled.c_simpleArray.simpleArray'>":
            return a
        elif str(type(a)) == "<type 'numpy.ndarray'>":
            rs = np.array([])
            arr = simpleArray()
            arr.createFromNumpy(a, rs)
            return arr
        else:
            raise ValueError("TrainData: convertFromSourceFile MUST produce either a list of numpy arrays or a list of DeepJetCore simpleArrays!")
            
    def _store(self, x, y, w):
        for xa in x:
            self.storeFeatureArray(self._maybeConvertToSimpleArray(xa))
        x = [] #collect garbage
        for ya in y:
            self.storeTruthArray(self._maybeConvertToSimpleArray(ya))
        y = []
        for wa in w:
            self.storeWeightArray(self._maybeConvertToSimpleArray(wa))
        w = []    
        
    def readFromSourceFile(self,filename, weighterobjects={}, istraining=False):
        x,y,w = self.convertFromSourceFile(filename, weighterobjects, istraining)
        self._store(x,y,w)
        

    ################# functions to be defined by the user    
        
    def createWeighterObjects(self, allsourcefiles):
        '''
        Will be called on the full list of source files once.
        Can be used to create weighter objects or similar that can
        then be applied to each individual conversion.
        Should return a dictionary
        '''
        return {}
    
    ### either of the following need to be defined
    
    ## if direct writeout is useful
    def writeFromSourceFile(self, filename, weighterobjects, istraining, outname):
        self.readFromSourceFile(filename, weighterobjects, istraining)
        self.writeToFile(outname)
    
    ## otherwise only define the conversion rule
    # returns a list of numpy arrays OR simpleArray (mandatory for ragged tensors)
    def convertFromSourceFile(self, filename, weighterobjects, istraining):
        return [],[],[]
    
    ## defines how to write out the prediction
    def writeOutPrediction(self, predicted, features, truth, weights, outfilename, inputfile):
        pass

