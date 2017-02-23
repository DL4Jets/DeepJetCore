'''
Created on 23 Feb 2017

@author: jkiesele
'''
from TrainData import TrainData

class TrainData_templ(TrainData):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
        TrainData.__init__(self)
        
    def readFromRootFile(self,filename,means):
        import numpy
        Tuple = self.readTreeFromRootToTuple(filename)
        
        weights=numpy.array([])
        x_all=numpy.array([])
        labels=numpy.array([])
        
        self.w=[weights]
        self.x=[x_all]
        self.y=[labels]
        