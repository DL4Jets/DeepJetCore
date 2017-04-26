from TrainData import TrainData, fileTimeOut
from TrainData import TrainData_simpleTruth
import numpy
from pdb import set_trace

class TrainData_deepCSV_PF_Binned(TrainData_simpleTruth):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData.__init__(self)

        #class specific
        self.charged_per_bin = 5 # number of charged particles per 2D bin
        self.neutral_per_bin = 5 # number of charged particles per 2D bin
        self.nbins = 20
        self.jet_radius = 0.4
        
        self.addBranches(['jet_pt', 'jet_eta','nCpfcand','nNpfcand','nsv','npv'])
       
        self.addBranches(['Cpfcan_BtagPf_trackEtaRel',
                          'Cpfcan_BtagPf_trackPtRel',
                          'Cpfcan_BtagPf_trackDeltaR',
                          'Cpfcan_BtagPf_trackPtRatio',
                          'Cpfcan_BtagPf_trackSip2dSig',
                          'Cpfcan_BtagPf_trackSip3dSig',
                          'Cpfcan_BtagPf_trackJetDistVal',
                          'Cpfcan_BtagPf_trackJetDistSig',
                          
                          'Cpfcan_erel',
                          'Cpfcan_drminsv',
                          'Cpfcan_chi2',
                          'Cpfcan_fromPV',
                          'Cpfcan_VTX_ass'
                              ],
                             20)
        
        
        self.addBranches(['Npfcan_erel',
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

        
        
       
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        from preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles, MeanNormZeroPadBinned
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
        
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        # Needed
        # (dimension #1, center #1, nbins 1, half width 1)
        # (dimension #2, center #2, nbins 2, half width 2)
        # sum o stack -- max to stack/zero pad        
        x_cpf = MeanNormZeroPadBinned(
            filename, TupleMeanStd, self.branches[1], 
            self.charged_per_bin, #per-bin # of particles to be kept. Hardoced here for the moment FIXME
            self.nsamples, 
            ('Cpfcan_eta', 'jet_eta', self.nbins, self.jet_radius), 
            ('Cpfcan_phi', 'jet_phi', self.nbins, self.jet_radius), 
        )
        
        x_npf = MeanNormZeroPadBinned(
            filename, TupleMeanStd, self.branches[1], 
            self.charged_per_bin, #per-bin # of particles to be kept. Hardoced here for the moment FIXME
            self.nsamples, 
            ('Npfcan_eta', 'jet_eta', self.nbins, self.jet_radius), 
            ('Npfcan_phi', 'jet_phi', self.nbins, self.jet_radius), 
        )
        
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        
        
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        if self.remove:
            notremoves=weighter.createNotRemoveIndices(Tuple)
            undef=Tuple['isUndefined']
            notremoves-=undef
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
