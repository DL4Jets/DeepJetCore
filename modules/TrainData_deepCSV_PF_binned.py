from TrainData import TrainData, fileTimeOut
from TrainData import TrainData_simpleTruth
import numpy
from pdb import set_trace

#
# TODO: add track and neutral pt in each bin (dz?, PUPPI value?, #hits? all the things for trk selection)
# Add general bin invo (charged pt, charged eta, # charged, # neutral, # svs 
#

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
        self.neutral_per_bin = 5 # number of charged particles per 2D bin
        self.nbins = 9
        self.jet_radius = 0.6
        self.regtruth='gen_pt_WithNu'        
        self.registerBranches([self.regtruth])
        
        self.addBranches(['jet_pt', 'jet_eta', 'nCpfcand', 'nNpfcand', 'nsv', 'rho'])
       
        self.addBranches([
            #B-tagging stuff
            'Cpfcan_BtagPf_trackEtaRel',
            'Cpfcan_BtagPf_trackPtRel',
            'Cpfcan_BtagPf_trackDeltaR',
            'Cpfcan_BtagPf_trackPtRatio',
            'Cpfcan_BtagPf_trackSip2dSig',
            'Cpfcan_BtagPf_trackSip3dSig',
            'Cpfcan_BtagPf_trackJetDistVal',
            'Cpfcan_BtagPf_trackJetDistSig',
            #Additional stuff
            'Cpfcan_erel',
            'Cpfcan_drminsv',
            'Cpfcan_chi2',
            'Cpfcan_fromPV',
            'Cpfcan_VTX_ass',
            #track selection
            'Cpfcan_dxy',
            'Cpfcan_dz',
            #for regression
            'Cpfcan_pt',
        ], 3)
        
        
        self.addBranches([
            #for b-tagging
            'Npfcan_erel',
            'Npfcan_deltaR',
            'Npfcan_isGamma',
            'Npfcan_HadFrac',
            'Npfcan_drminsv',
            #for regression
            'Npfcan_pt',
        ], 2)
        
        
        self.addBranches([
            'sv_pt',
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
        ], 1)

        self.binned_sums = {
            'charged' : ['Cpfcan_pt'], #of objects already set by default
            'neutral' : ['Npfcan_pt'], #of objects already set by default
            'svs'     : [],
            }

        self.sums_scaling = {
            'charged' : ['nCpfcand', 'jet_pt'], #of objects already set by default
            'neutral' : ['nNpfcand', 'jet_pt'], #of objects already set by default
            'svs'     : ['nsv'],
            }
        
    def getTruthShapes(self):
        outl = [len(self.getClasses()), 1]
        return outl
        
    def getClasses(self):
        if len(self.reducedtruthclasses) > 0:
            return self.reducedtruthclasses
        else:
            return self.truthclasses
        
       
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
        #self.nsamples = 10 #TESTING

        print('took ', sw.getAndReset(), ' seconds for getting tree entries')
        
        # split for convolutional network
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        # needed
        # (dimension #1, center #1, nbins 1, half width 1)
        # (dimension #2, center #2, nbins 2, half width 2)
        # sum o stack -- max to stack/zero pad        
        x_cpf, sum_cpf = MeanNormZeroPadBinned(
            filename, 'nCpfcand', self.nsamples,
            ('Cpfcan_eta', 'jet_eta', self.nbins, self.jet_radius), #X axis
            ('Cpfcan_phi', 'jet_phi', self.nbins, self.jet_radius), #Y axis
            (TupleMeanStd, self.branches[1], self.branchcutoffs[1]), #means/std, branches to use, #per-bin # of particles to be kept            
            (self.sums_scaling['charged'], self.binned_sums['charged']), #variables to be summed (no zero padding yet)
        )

        x_npf, sum_npf = MeanNormZeroPadBinned(
            filename, 'nNpfcand', self.nsamples,
            ('Npfcan_eta', 'jet_eta', self.nbins, self.jet_radius), 
            ('Npfcan_phi', 'jet_phi', self.nbins, self.jet_radius), 
            (TupleMeanStd, self.branches[2], self.branchcutoffs[2]),
            (self.sums_scaling['neutral'], self.binned_sums['neutral']),
        )
        
        x_sv, sum_sv = MeanNormZeroPadBinned(
            filename, 'nsv', self.nsamples, 
            ('sv_eta', 'jet_eta', self.nbins, self.jet_radius), 
            ('sv_phi', 'jet_phi', self.nbins, self.jet_radius), 
            (TupleMeanStd, self.branches[3], self.branchcutoffs[3]),
            (self.sums_scaling['svs'], self.binned_sums['svs']),
        )

        #merging sum variables together
        x_sum = numpy.concatenate((sum_cpf, sum_npf, sum_sv), axis=3)

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
        pt_truth = Tuple[self.regtruth]
        
        #print(alltruth.shape)
        if self.remove:
            print('remove')
            weights = weights[notremoves > 0]
            x_global = x_global[notremoves > 0]
            x_cpf = x_cpf[notremoves > 0]
            x_npf = x_npf[notremoves > 0]
            x_sv  = x_sv[notremoves > 0]
            x_sum = x_sum[notremoves > 0]
            alltruth = alltruth[notremoves > 0]
            pt_truth = pt_truth[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        self.w = [weights]
        self.x = [x_global, x_cpf, x_npf, x_sv, x_sum]
        self.y = [alltruth, pt_truth]
        
