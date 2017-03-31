'''
Created on 21 Feb 2017

@author: jkiesele
'''

from TrainData import TrainData
import numpy

class TrainData_deepCSV_PF(TrainData):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        TrainData.__init__(self)
        
        self.truthclasses=['isB','isBB','isLeptonicB','isLeptonicB_C','isC','isUDS','isG']
        
        self.addBranches(['jet_pt', 'jet_eta','nCpfcand','nNpfcand','nsv','npv'])
       
        self.addBranches(['Cpfcan_pt',
                          'Cpfcan_ptrel',
                          'Cpfcan_erel',
                              'Cpfcan_phirel',
                              'Cpfcan_etarel', 
                              'Cpfcan_deltaR', 
                              'Cpfcan_puppiw',
                              'Cpfcan_dxy', 
                              
                              'Cpfcan_dxyerr', 
                              'Cpfcan_dxysig', 
                              
                              'Cpfcan_dz', 
                              'Cpfcan_VTX_ass', 
                              'Cpfcan_fromPV', 
                              'Cpfcan_drminsv', 
                              
                              'Cpfcan_vertex_rho', 
                              'Cpfcan_vertex_phirel', 
                              'Cpfcan_vertex_etarel',
                              
                              'Cpfcan_dptdpt', 
                              'Cpfcan_detadeta',
                              'Cpfcan_dphidphi',
                              
                              'Cpfcan_dxydxy',
                              'Cpfcan_dzdz',
                              'Cpfcan_dxydz',
                              'Cpfcan_dphidxy',
                              'Cpfcan_dlambdadz',
                              
                              #'Cpfcan_isMu',
                              #'Cpfcan_isEl',
                              'Cpfcan_chi2',
                              'Cpfcan_quality'
                              ],
                             20)
        
        
        self.addBranches(['Npfcan_pt',
                          'Npfcan_ptrel',
                          'Npfcan_erel',
                          
                          'Npfcan_phirel',
                          'Npfcan_etarel',
                          'Npfcan_deltaR',
                              'Npfcan_isGamma',
                              'Npfcan_HadFrac',
                              'Npfcan_drminsv',
                              ],
                             15)
        
        
        self.addBranches(['sv_pt',
                              'sv_etarel',
                              'sv_phirel',
                              'sv_deltaR',
                              'sv_mass',
                              'sv_ntracks',
                              'sv_chi2',
                              'sv_ndf',
                              'sv_normchi2',
                              'sv_dxy',
                              'sv_dxyerr',
                              'sv_dxysig',
                              'sv_d3d',
                              'sv_d3derr',
                              'sv_d3dsig',
                              'sv_costhetasvpv',
                              'sv_enratio',
                              ],
                             4)

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
       
       
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        from preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles
        import numpy
        from stopwatch import stopwatch
        
        sw=stopwatch()
        swall=stopwatch()
        
        import ROOT
        
        self.fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples=tree.GetEntries()
        
        print('took ', sw.getAndReset(), ' seconds for getting tree entries')
        
        
        # split for convolutional network
        
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        x_npf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        
        
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        if self.remove:
            notremoves=weighter.createNotRemoveIndices(Tuple)
        
            print('took ', sw.getAndReset(), ' to create remove indices')
        
        if self.weight:
            weights=weighter.getJetWeights(Tuple)
        elif self.remove:
            weights=notremoves
        else:
            print('neither remove nor weight')
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_sv=x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights]
        self.x=[x_global,x_cpf,x_npf,x_sv]
        self.y=[alltruth]
        
        

class TrainData_deepCSV_miniPF(TrainData_deepCSV_PF):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        TrainData.__init__(self)
        
        self.truthclasses=['isB','isBB','isLeptonicB','isLeptonicB_C','isC','isUDS','isG']
        
        self.addBranches(['jet_pt', 'jet_eta','nCpfcand','nNpfcand','nsv','npv'])
       
        self.addBranches([#'Cpfcan_pt',
                          'Cpfcan_ptrel',
                          #'Cpfcan_erel',
                              #'Cpfcan_phirel',
                              #'Cpfcan_etarel', 
                              'Cpfcan_deltaR', 
                              #'Cpfcan_puppiw',
                              'Cpfcan_dxy', 
                              
                              #'Cpfcan_dxyerr', 
                              'Cpfcan_dxysig', 
                              
                              #'Cpfcan_dz', 
                              #'Cpfcan_VTX_ass', 
                              'Cpfcan_fromPV', 
                              'Cpfcan_drminsv', 
                              
                              'Cpfcan_vertex_rho', 
                              'Cpfcan_vertex_phirel', 
                              'Cpfcan_vertex_etarel',
                              
                              #'Cpfcan_dptdpt', 
                              'Cpfcan_detadeta',
                              'Cpfcan_dphidphi',
                              
                              'Cpfcan_dxydxy',
                              'Cpfcan_dzdz',
                              'Cpfcan_dxydz',
                              'Cpfcan_dphidxy',
                              #'Cpfcan_dlambdadz',
                              
                              #'Cpfcan_isMu',
                              #'Cpfcan_isEl',
                              'Cpfcan_chi2',
                              #'Cpfcan_quality'
                              ],
                             15)
        
        
        self.addBranches([#'Npfcan_pt',
                          'Npfcan_ptrel',
                          #'Npfcan_erel',
                          
                          #'Npfcan_phirel',
                          #'Npfcan_etarel',
                          'Npfcan_deltaR',
                              #'Npfcan_isGamma',
                              'Npfcan_HadFrac',
                              'Npfcan_drminsv',
                              ],
                             15)
        
        
        self.addBranches(['sv_pt',
                              #'sv_etarel',
                              #'sv_phirel',
                              'sv_deltaR',
                              'sv_mass',
                              'sv_ntracks',
                              #'sv_chi2',
                              #'sv_ndf',
                              'sv_normchi2',
                              'sv_dxy',
                              #'sv_dxyerr',
                              'sv_dxysig',
                              'sv_d3d',
                              #'sv_d3derr',
                              'sv_d3dsig',
                              'sv_costhetasvpv',
                              'sv_enratio',
                              ],
                             4)
        
        #b-tag vars
        self.addBranches([ 'TagVarCSV_jetNSecondaryVertices', 
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
                             7)
        
        
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


    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        from preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles
        import numpy
        from stopwatch import stopwatch
        
        sw=stopwatch()
        swall=stopwatch()
        
        import ROOT
        
        self.fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples=tree.GetEntries()
        
        print('took ', sw.getAndReset(), ' seconds for getting tree entries')
        
        
        # split for convolutional network
        
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        x_npf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        
        
        
        x_global_csv = MeanNormZeroPad(filename,TupleMeanStd,
                                   self.branches[4:],
                                   self.branchcutoffs[4:],self.nsamples)
        
        
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        if self.remove:
            notremoves=weighter.createNotRemoveIndices(Tuple)
        
            print('took ', sw.getAndReset(), ' to create remove indices')
        
        if self.weight:
            weights=weighter.getJetWeights(Tuple)
        elif self.remove:
            weights=notremoves
        else:
            print('neither remove nor weight')
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_npf=x_npf[notremoves > 0]
            x_sv=x_sv[notremoves > 0]
            x_global_csv=x_global_csv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights]
        self.x=[x_global,x_cpf,x_npf,x_sv,x_global_csv]
        self.y=[alltruth]