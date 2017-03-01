'''
Created on 21 Feb 2017

@author: jkiesele
'''
from TrainData import TrainData_Flavour
import numpy

class TrainData_deepCSV(TrainData_Flavour):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        TrainData_Flavour.__init__(self)
        
        self.truthclasses=['isB','isC','isUDS','isG']
        
        self.flatbranches=['jet_pt', 'jet_eta',
                           'TagVarCSV_jetNSecondaryVertices', 
                           'TagVarCSV_trackSumJetEtRatio', 'TagVarCSV_trackSumJetDeltaR', 
                           'TagVarCSV_vertexCategory', 'TagVarCSV_trackSip2dValAboveCharm', 
                           'TagVarCSV_trackSip2dSigAboveCharm', 'TagVarCSV_trackSip3dValAboveCharm', 
                           'TagVarCSV_trackSip3dSigAboveCharm', 'TagVarCSV_jetNSelectedTracks', 
                           'TagVarCSV_jetNTracksEtaRel']
       
        self.addDeepBranches(['TagVarCSVTrk_trackJetDistVal',
                              'TagVarCSVTrk_trackPtRel', 
                              'TagVarCSVTrk_trackDeltaR', 
                              'TagVarCSVTrk_trackPtRatio', 
                              'TagVarCSVTrk_trackSip3dSig', 
                              'TagVarCSVTrk_trackSip2dSig', 
                              'TagVarCSVTrk_trackDecayLenVal'],
                             10)
        
        
        self.addDeepBranches(['TagVarCSV_trackEtaRel'],8)

        self.addDeepBranches(['TagVarCSV_vertexMass', 
                              'TagVarCSV_vertexNTracks', 
                              'TagVarCSV_vertexEnergyRatio',
                              'TagVarCSV_vertexJetDeltaR',
                              'TagVarCSV_flightDistance2dVal', 
                              'TagVarCSV_flightDistance2dSig', 
                              'TagVarCSV_flightDistance3dVal', 
                              'TagVarCSV_flightDistance3dSig'],
                             5)
    
    
    def reduceTruth(self, tuple_in):
        b = tuple_in['isB'].view(numpy.ndarray)
        c = tuple_in['isC'].view(numpy.ndarray)
        uds = tuple_in['isUDS'].view(numpy.ndarray)
        g = tuple_in['isG'].view(numpy.ndarray)
        l = g + uds
        self.truthclasses=['isB','isC','isUDSG']
        return numpy.vstack((b,c,l)).transpose()
       
