'''
Created on 21 Feb 2017

@author: jkiesele
'''
from TrainData import TrainData_Flavour


class TrainData_deepCSV(TrainData_Flavour):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        TrainData_Flavour.__init__(self)
        
        self.truthclasses=['isB','isBB','isLeptonicB','isLeptonicB_C','isC','isUDS','isG']
        
        self.addBranches(['jet_pt', 'jet_eta',
                           'TagVarCSV_jetNSecondaryVertices', 
                           'TagVarCSV_trackSumJetEtRatio', 'TagVarCSV_trackSumJetDeltaR', 
                           'TagVarCSV_vertexCategory', 'TagVarCSV_trackSip2dValAboveCharm', 
                           'TagVarCSV_trackSip2dSigAboveCharm', 'TagVarCSV_trackSip3dValAboveCharm', 
                           'TagVarCSV_trackSip3dSigAboveCharm', 'TagVarCSV_jetNSelectedTracks', 
                           'TagVarCSV_jetNTracksEtaRel'])
       
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
        
        self.reducedtruthclasses=['isB','isBB','isC','isUDSG']
    
    def reduceTruth(self, tuple_in):
        import numpy
        b = tuple_in['isB'].view(numpy.ndarray)
        bb = tuple_in['isBB'].view(numpy.ndarray)
        bl = tuple_in['isLeptonicB'].view(numpy.ndarray)
        blc = tuple_in['isLeptonicB_C'].view(numpy.ndarray)
        
        c = tuple_in['isC'].view(numpy.ndarray)
        uds = tuple_in['isUDS'].view(numpy.ndarray)
        g = tuple_in['isG'].view(numpy.ndarray)
        allb = b+bl+blc
        l = g + uds
        self.reducedtruthclasses=['isB','isBB','isC','isUDSG']
        return numpy.vstack((allb,bb,c,l)).transpose()
        
    
