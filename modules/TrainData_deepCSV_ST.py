'''
Created on 21 Feb 2017

@author: jkiesele
'''
from TrainData import TrainData_Flavour

import numpy


class TrainData_deepCSV_ST(TrainData_Flavour):
    '''
    same as TrainData_deepCSV but with 3 truth labels: UDSG C B
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


        self.reducedtruthclasses=['isB','isC','isUDSG']
        
    def reduceTruth(self, tuple_in):
        b = tuple_in['isB'].view(numpy.ndarray)
        bb = tuple_in['isBB'].view(numpy.ndarray)
        bl = tuple_in['isLeptonicB'].view(numpy.ndarray)
        blc = tuple_in['isLeptonicB_C'].view(numpy.ndarray)
        
        c = tuple_in['isC'].view(numpy.ndarray)
        uds = tuple_in['isUDS'].view(numpy.ndarray)
        g = tuple_in['isG'].view(numpy.ndarray)
        allb = b+bb+bl+blc
        l = g + uds
        self.reducedtruthclasses=['isB','isC','isUDSG']
        return numpy.vstack((allb,c,l)).transpose()
        

class TrainData_deepCMVA_SST(TrainData_Flavour):
    '''
    same as TrainData_deepCSV but with 3 truth labels: UDSG C B
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


        self.reducedtruthclasses=['isB','isC','isUDSG']
        
    def reduceTruth(self, tuple_in):
        b = tuple_in['isB'].view(numpy.ndarray)
        bb = tuple_in['isBB'].view(numpy.ndarray)
        bl = tuple_in['isLeptonicB'].view(numpy.ndarray)
        blc = tuple_in['isLeptonicB_C'].view(numpy.ndarray)
        
        c = tuple_in['isC'].view(numpy.ndarray)
        uds = tuple_in['isUDS'].view(numpy.ndarray)
        g = tuple_in['isG'].view(numpy.ndarray)
        allb = b+bb+bl+blc
        l = g + uds
        self.reducedtruthclasses=['isB','isC','isUDSG']
        return numpy.vstack((allb,c,l)).transpose()
        
        
                
class TrainData_deepCMVA_ST(TrainData_Flavour):
    '''
    same as TrainData_deepCSV but with 5 truth labels: UDSG C B leptonicB leptonicB_C
    '''


    def __init__(self):
        '''
        Constructor
        '''
        TrainData_Flavour.__init__(self)
        self.truthclasses=['isB','isLeptonicB','isLeptonicB_C','isC','isUDS','isG']
        
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


        self.reducedtruthclasses=['isB','isLeptonicB','isLeptonicB_C','isC','isUDSG']
        
    def reduceTruth(self, tuple_in):
        b = tuple_in['isB'].view(numpy.ndarray)
        lep_b = tuple_in['isLeptonicB'].view(numpy.ndarray)
        lep_b_c = tuple_in['isLeptonicB_C'].view(numpy.ndarray)
        c = tuple_in['isC'].view(numpy.ndarray)
        uds = tuple_in['isUDS'].view(numpy.ndarray)
        g = tuple_in['isG'].view(numpy.ndarray)
        l = g + uds
        self.reducedtruthclasses=['isB','isLeptonicB','isLeptonicB_C','isC','isUDSG']
        return numpy.vstack((b,lep_b,lep_b_c,c,l)).transpose()


class TrainData_deepCSV_ST_broad(TrainData_Flavour):
    '''
    same as TrainData_deepCSV but with 3 truth labels: UDSG C B
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
                             10)
        
        
        self.addBranches(['TagVarCSV_trackEtaRel'],8)

        self.addBranches(['TagVarCSV_vertexMass', 
                              'TagVarCSV_vertexNTracks', 
                              'TagVarCSV_vertexEnergyRatio',
                              'TagVarCSV_vertexJetDeltaR',
                              'TagVarCSV_flightDistance2dVal', 
                              'TagVarCSV_flightDistance2dSig', 
                              'TagVarCSV_flightDistance3dVal', 
                              'TagVarCSV_flightDistance3dSig'],
                             5)


        self.reducedtruthclasses=['isB','isC','isUDSG']
        
    def reduceTruth(self, tuple_in):
        b = tuple_in['isB'].view(numpy.ndarray)
        bb = tuple_in['isBB'].view(numpy.ndarray)
        bl = tuple_in['isLeptonicB'].view(numpy.ndarray)
        blc = tuple_in['isLeptonicB_C'].view(numpy.ndarray)
        
        c = tuple_in['isC'].view(numpy.ndarray)
        uds = tuple_in['isUDS'].view(numpy.ndarray)
        g = tuple_in['isG'].view(numpy.ndarray)
        allb = b+bb+bl+blc
        l = g + uds
        self.reducedtruthclasses=['isB','isC','isUDSG']
        return numpy.vstack((allb,c,l)).transpose()
        
        
      
      
