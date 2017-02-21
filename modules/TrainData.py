'''
Created on 20 Feb 2017

@author: jkiesele
'''


import numpy

class TrainData(object):
    '''
    Base class for batch-wise training of the DNN
    '''
    def __init__(self):
        '''
        Constructor
        '''
        import numpy as np
        
        self.x=np.array([])
        self.y=np.array([])
        self.w=np.array([])
        
    def clear(self):
        import numpy as np
        self.x=np.array([])
        self.y=np.array([])
        self.w=np.array([])
        
    def addFromRootFile(self,fileName):
        '''
        Adds from a root file and randomly shuffles the input
        '''
        raise Exception('to be implemented')
        #just call read from root (virtual in python??), and mix with existing x,y,weight

    def writeOut(self,fileprefix):
        
        numpy.save(fileprefix+"_w0.npy",self.w[0])
        numpy.save(fileprefix+"_w1.npy",self.w[1])
        numpy.save(fileprefix+"_x0.npy",self.x[0])
        numpy.save(fileprefix+"_x1.npy",self.x[1])
        numpy.save(fileprefix+"_y0.npy",self.y[0])
        numpy.save(fileprefix+"_y1.npy",self.y[1])
        
    def readIn(self,fileprefix):
        #probably not necessary
        self.w=[[],[]]
        self.x=[[],[]]
        self.y=[[],[]]
        self.w[0] = numpy.load(fileprefix+"_w0.npy")
        self.w[1] = numpy.load(fileprefix+"_w1.npy")
        self.x[0] = numpy.load(fileprefix+"_x0.npy")
        self.x[1] = numpy.load(fileprefix+"_x1.npy")
        self.y[0] = numpy.load(fileprefix+"_y0.npy")
        self.y[1] = numpy.load(fileprefix+"_y1.npy")
        
        