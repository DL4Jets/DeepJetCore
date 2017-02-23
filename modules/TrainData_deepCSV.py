'''
Created on 21 Feb 2017

@author: jkiesele
'''
from TrainData import TrainData

import numpy
import ROOT
from root_numpy import tree2array
from preprocessing import produceWeigths, meanNormProd, MakeBox, MeanNormApply, MeanNormZeroPad

class TrainData_deepCSV(TrainData):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        TrainData.__init__(self)
    
    def readFromRootFile(self,filename):
        
        # may want to split this to a more generic function to allow shuffeling later
        # maybe something like "addfromRootFile" -> should go to base class
        
        Tuple = self.readTreeFromRootToTuple(filename)

        TupleMeanStd =  meanNormProd(Tuple) #only for first then apply to all
        
        # sanity checks, would brake easily if wrong means and std are used (dimension check)
        BranchList = Tuple.dtype.names
        if BranchList != TupleMeanStd.dtype.names:
            print ('Tuple for subtraction and training should match, please check')
            print (len(BranchList), ' ' , len(BranchList))
        #print (BranchList)
        
        # now we calculate weights to have flat PT eta distributions
        # entries per bin (not x-section, i.e. entries/density) will be flattened
        weight_binXPt = numpy.array([10,25,30,35,40,45,50,60,75,2000],dtype=float)
        weight_binYEta = numpy.array([0,.4,.8,1.2,1.6,2.,2.4],dtype=float)
        weights = produceWeigths(Tuple,"jet_pt","jet_eta",[weight_binXPt,weight_binYEta],classes=['isB','isC','isUDS','isG'])
        # dimension check, weight vector must have tuple length
        if weights.shape[0] != Tuple.shape[0]:
            print ('Weigts for subtraction and training should match, please check')
            print  (weights.shape[0],' ', Tuple.shape[0])
        
       
        flatBranches = ['jet_pt', 'jet_eta','TagVarCSV_jetNSecondaryVertices', 'TagVarCSV_trackSumJetEtRatio', 'TagVarCSV_trackSumJetDeltaR', 'TagVarCSV_vertexCategory', 'TagVarCSV_trackSip2dValAboveCharm', 'TagVarCSV_trackSip2dSigAboveCharm', 'TagVarCSV_trackSip3dValAboveCharm', 'TagVarCSV_trackSip3dSigAboveCharm', 'TagVarCSV_jetNSelectedTracks', 'TagVarCSV_jetNTracksEtaRel']
        tracksBranches = ['TagVarCSVTrk_trackJetDistVal','TagVarCSVTrk_trackPtRel', 'TagVarCSVTrk_trackDeltaR', 'TagVarCSVTrk_trackPtRatio', 'TagVarCSVTrk_trackSip3dSig', 'TagVarCSVTrk_trackSip2dSig', 'TagVarCSVTrk_trackDecayLenVal']
        tracksEtaRel = ['TagVarCSV_trackEtaRel']
        sv = ['TagVarCSV_vertexMass', 'TagVarCSV_vertexNTracks', 'TagVarCSV_vertexEnergyRatio','TagVarCSV_vertexJetDeltaR','TagVarCSV_flightDistance2dVal', 'TagVarCSV_flightDistance2dSig', 'TagVarCSV_flightDistance3dVal', 'TagVarCSV_flightDistance3dSig']
        
        x_global_flat = MeanNormApply(Tuple[flatBranches],TupleMeanStd)
        x_tracks = MeanNormZeroPad(Tuple[tracksBranches],TupleMeanStd,6)
        x_tracksEtaRel = MeanNormZeroPad(Tuple[tracksEtaRel],TupleMeanStd,4)
        x_sv = MeanNormZeroPad(Tuple[sv],TupleMeanStd,1)
        #print(x_global_flat.shape , x_tracks.shape,' ' , x_tracksEtaRel.shape, ' ', x_sv.shape)
        # make to an narray
        x_global_flat = numpy.array(x_global_flat.tolist())
        x_all = numpy.concatenate( (x_global_flat,x_tracks,x_tracksEtaRel,x_sv) , axis=1)

        Flavour_truth =  Tuple[['isB','isC','isUDS','isG']]
        
        #merge UDS G
        
        
        
        #####needs to be filled in any implementation
        
        self.w=[weights]
        self.x=[x_all]
        self.y=[numpy.array(Flavour_truth.tolist())]
        
       
