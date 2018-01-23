'''
Created on 21 Feb 2017

@author: jkiesele
'''
from TrainData import TrainData_Flavour, TrainData_simpleTruth, TrainData_fullTruth, fileTimeOut



class TrainData_deepCSV(TrainData_Flavour, TrainData_simpleTruth):
    '''
    same as TrainData_deepCSV but with 4 truth labels: B BB C UDSG
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

    def readFromRootFile(self,filename,TupleMeanStd, weighter):
			super(TrainData_deepCSV, self).readFromRootFile(filename, TupleMeanStd, weighter)
			ys = self.y[0]
			flav_sum = ys.sum(axis=1)
			if (flav_sum > 1).any():
				raise ValueError('In file: %s I get a jet with multiple flavours assigned!' % filename)
			mask = (flav_sum == 1)
			self.x = [self.x[0][mask]]
			self.y = [self.y[0][mask]]
			self.w = [self.w[0][mask]]

class TrainData_deepCSV_RNN(TrainData_fullTruth):
    '''
    same as TrainData_deepCSV but with 4 truth labels: B BB C UDSG
    '''
    def __init__(self):
        '''
        Constructor
        '''
        super(TrainData_deepCSV_RNN, self).__init__()
        
        self.addBranches([
            'jet_pt', 'jet_eta',
            'TagVarCSV_jetNSecondaryVertices', 
            'TagVarCSV_trackSumJetEtRatio', 'TagVarCSV_trackSumJetDeltaR', 
            'TagVarCSV_vertexCategory', 'TagVarCSV_trackSip2dValAboveCharm', 
            'TagVarCSV_trackSip2dSigAboveCharm', 'TagVarCSV_trackSip3dValAboveCharm', 
            'TagVarCSV_trackSip3dSigAboveCharm', 'TagVarCSV_jetNSelectedTracks', 
            'TagVarCSV_jetNTracksEtaRel'])
       
        self.addBranches([
            'TagVarCSVTrk_trackJetDistVal',
            'TagVarCSVTrk_trackPtRel', 
            'TagVarCSVTrk_trackDeltaR', 
            'TagVarCSVTrk_trackPtRatio', 
            'TagVarCSVTrk_trackSip3dSig', 
            'TagVarCSVTrk_trackSip2dSig', 
            'TagVarCSVTrk_trackDecayLenVal'
        ], 6)
        
        
        self.addBranches(['TagVarCSV_trackEtaRel'],4)

        self.addBranches([
            'TagVarCSV_vertexMass', 
            'TagVarCSV_vertexNTracks', 
            'TagVarCSV_vertexEnergyRatio',
            'TagVarCSV_vertexJetDeltaR',
            'TagVarCSV_flightDistance2dVal', 
            'TagVarCSV_flightDistance2dSig', 
            'TagVarCSV_flightDistance3dVal', 
            'TagVarCSV_flightDistance3dSig'
        ], 1)

        self.addBranches(['jet_corr_pt'])
        self.registerBranches(['gen_pt_WithNu'])
        self.regressiontargetclasses=['uncPt','Pt']


    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        from preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles
        import numpy
        from stopwatch import stopwatch
        
        sw=stopwatch()
        swall=stopwatch()
        
        import ROOT
        
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples=tree.GetEntries()
        
        print('took ', sw.getAndReset(), ' seconds for getting tree entries')
        
        
        # split for convolutional network
        
        x_global = MeanNormZeroPad(
            filename,None,
            [self.branches[0]],
            [self.branchcutoffs[0]],self.nsamples
        )
        
        x_cpf = MeanNormZeroPadParticles(
            filename,None,
            self.branches[1],
            self.branchcutoffs[1],self.nsamples
        )
        
        x_etarel = MeanNormZeroPadParticles(
            filename,None,
            self.branches[2],
            self.branchcutoffs[2],self.nsamples
        )
        
        x_sv = MeanNormZeroPadParticles(
            filename,None,
            self.branches[3],
            self.branchcutoffs[3],self.nsamples
        )
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        npy_array = self.readTreeFromRootToTuple(filename)
        
        reg_truth=npy_array['gen_pt_WithNu'].view(numpy.ndarray)
        reco_pt=npy_array['jet_corr_pt'].view(numpy.ndarray)
        
        correctionfactor=numpy.zeros(self.nsamples)
        for i in range(self.nsamples):
            correctionfactor[i]=reg_truth[i]/reco_pt[i]

        truthtuple =  npy_array[self.truthclasses]
        alltruth=self.reduceTruth(truthtuple)
        
        self.x=[x_global, x_cpf, x_etarel, x_sv, reco_pt]
        self.y=[alltruth,correctionfactor]
        self._normalize_input_(weighter, npy_array)

        
    
    

class TrainData_deepCSV_RNN_Deeper(TrainData_deepCSV_RNN):
    '''
    same as TrainData_deepCSV but with 4 truth labels: B BB C UDSG
    '''
    def __init__(self):
        '''
        Constructor
        '''
        super(TrainData_deepCSV_RNN_Deeper, self).__init__()
        self.branchcutoffs = [1, 20, 13, 4, 1]

