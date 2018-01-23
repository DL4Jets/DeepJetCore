'''
Created on 21 Feb 2017

@author: jkiesele
'''
from TrainData import TrainData_Flavour, TrainData_simpleTruth, TrainData_fullTruth, fileTimeOut



class TrainData_sampleCheck(TrainData_Flavour, TrainData_simpleTruth):
    '''
    Simple train data to check we are not introducing and bias due to sample
		differences
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_Flavour.__init__(self)        
        self.addBranches(['jet_pt', 'jet_eta', 'jet_phi'])


