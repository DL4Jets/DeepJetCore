'''
Created on 21 Feb 2017

@author: jkiesele
'''
from TrainData import TrainData_Flavour, TrainData_leptTruth

class TrainData_deepCMVA(TrainData_Flavour,TrainData_leptTruth):
    '''
    same as TrainData_deepCSV but with 5 truth labels: B leptonicB leptonicB_C C UDSG
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_Flavour.__init__(self)
        
        self.addBranches(['jet_pt', 'jet_eta',
                           'TagVarCSV_jetNSecondaryVertices', 
                           'TagVarCSV_trackSumJetEtRatio', 'TagVarCSV_trackSumJetDeltaR', 
                           'TagVarCSV_vertexCategory', 'TagVarCSV_trackSip2dValAboveCharm', 
                           'TagVarCSV_trackSip2dSigAboveCharm', 'TagVarCSV_trackSip3dValAboveCharm', 
                           'TagVarCSV_trackSip3dSigAboveCharm', 'TagVarCSV_jetNSelectedTracks', 
                           'TagVarCSV_jetNTracksEtaRel',
                           'softPFMuonBJetTags', 'softPFElectronBJetTags',
                           'pfJetBProbabilityBJetTags', 'pfJetProbabilityBJetTags'])

        self.addBranches(['TagVarCSVTrk_trackJetDistVal',
                              'TagVarCSVTrk_trackPtRel', 
                              'TagVarCSVTrk_trackDeltaR', 
                              'TagVarCSVTrk_trackPtRatio', 
                              'TagVarCSVTrk_trackSip3dSig', 
                              'TagVarCSVTrk_trackSip2dSig', 
                              'TagVarCSVTrk_trackDecayLenVal'],
                             6)
        
        self.addBranches(['TagVarCSV_trackEtaRel'],4)

        self.addBranches(['TagVarCSV_vertexMass', 
                              'TagVarCSV_vertexNTracks', 
                              'TagVarCSV_vertexEnergyRatio',
                              'TagVarCSV_vertexJetDeltaR',
                              'TagVarCSV_flightDistance2dVal', 
                              'TagVarCSV_flightDistance2dSig', 
                              'TagVarCSV_flightDistance3dVal', 
                              'TagVarCSV_flightDistance3dSig'],
                             1)
        
        

