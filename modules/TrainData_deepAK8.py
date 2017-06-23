
from TrainData import TrainData,fileTimeOut
import numpy


class TrainData_AK8Jet(TrainData):
    
    def __init__(self):
        '''
        This class is meant as a base class for the FatJet studies
        You will not need to edit it for trying out things
        '''
        TrainData.__init__(self)
        
        #define truth:
        self.undefTruth=['isUndefined']
        self.truthclasses=['fj_isLight','fj_isW','fj_isZ','fj_isH','fj_isTop']
        self.referenceclass='fj_isLight' ## used for pt reshaping

        self.registerBranches(self.truthclasses)
        self.registerBranches(['fj_pt','fj_eta'])

        self.weightbranchX='fj_pt'
        self.weightbranchY='fj_eta'

        self.weight_binX = numpy.array([
                10,25,30,35,40,45,50,60,75,100,
                125,150,175,200,250,300,400,500,
                600,2000],dtype=float)
        
        self.weight_binY = numpy.array(
            [-2.5,2.5],
            dtype=float
            )


        
        #this is only needed because the truth definitions are different from deepFlavour
        self.allbranchestoberead=[]
        self.registerBranches(self.undefTruth)
        self.registerBranches(self.truthclasses)
        
        
    ## categories to use for training     
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses=['fj_isLight','fj_isW','fj_isZ','fj_isH','fj_isTop']
        if tuple_in is not None:
            q = tuple_in['fj_isLight'].view(numpy.ndarray)
            w = tuple_in['fj_isTop'].view(numpy.ndarray)
            z = tuple_in['fj_isZ'].view(numpy.ndarray)
            h = tuple_in['fj_isH'].view(numpy.ndarray)
            t = tuple_in['fj_isW'].view(numpy.ndarray)
            
            return numpy.vstack((q,w,z,h,t)).transpose()  
        
        
        
#######################################
        
        
class TrainData_AK8Jet_init(TrainData_AK8Jet):
    
    def __init__(self):
        '''
        This is an example data format description for FatJet studies
        '''
        TrainData_AK8Jet.__init__(self)
        
        #example of how to register global branches
        self.addBranches(['fj_pt',
                          'fj_eta',
                          'fj_sdmass',
                          'fj_n_sdsubjets',
                          'fj_doubleb',
                          'fj_tau21',
                          'fj_tau32',
                          'npv',
                          'npfcands',
                          'ntracks',
                          'nsv'
                      ])
        

        #example of pf candidate branches
        self.addBranches(['pfcand_ptrel',
                          'pfcand_erel',
                          'pfcand_phirel',
                          'pfcand_etarel',
                          'pfcand_deltaR',
                          'pfcand_puppiw',
                          'pfcand_drminsv',
                          'pfcand_drsubjet1',
                          'pfcand_drsubjet2',
                          'pfcand_hcalFrac'
                         ],
                         100) 

        self.addBranches(['track_ptrel',     
                          'track_erel',     
                          'track_phirel',     
                          'track_etarel',     
                          'track_deltaR',
                          'track_drminsv',     
                          'track_drsubjet1',     
                          'track_drsubjet2',
                          'track_dz',     
                          'track_dzsig',     
                          'track_dxy',     
                          'track_dxysig',     
                          'track_normchi2',     
                          'track_quality',     
                          'track_dptdpt',     
                          'track_detadeta',     
                          'track_dphidphi',     
                          'track_dxydxy',     
                          'track_dzdz',     
                          'track_dxydz',     
                          'track_dphidxy',     
                          'track_dlambdadz',     
                          'trackBTag_EtaRel',     
                          'trackBTag_PtRatio',     
                          'trackBTag_PParRatio',     
                          'trackBTag_Sip2dVal',     
                          'trackBTag_Sip2dSig',     
                          'trackBTag_Sip3dVal',     
                          'trackBTag_Sip3dSig',     
                          'trackBTag_JetDistVal'
                         ],
                         60) 
        
        self.addBranches(['sv_ptrel',
                          'sv_erel',
                          'sv_phirel',
                          'sv_etarel',
                          'sv_deltaR',
                          'sv_pt',
                          'sv_mass',
                          'sv_ntracks',
                          'sv_normchi2',
                          'sv_dxy',
                          'sv_dxysig',
                          'sv_d3d',
                          'sv_d3dsig',
                          'sv_costhetasvpv'
                         ],
                         5)

        
        #branches that are used directly in the following function 'readFromRootFile'
        #this is a technical trick to speed up the conversion
        #self.registerBranches(['Cpfcan_erel','Cpfcan_eta','Cpfcan_phi',
        #                       'Npfcan_erel','Npfcan_eta','Npfcan_phi',
        #                       'nCpfcand','nNpfcand',
        #                       'jet_eta','jet_phi'])
        
        
    #this function describes how the branches are converted
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        
        #the first part is standard, no changes needed
        from preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles
        import numpy
        import ROOT
        
        fileTimeOut(filename,120) #give eos 2 minutes to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples=tree.GetEntries()
        
        #the definition of what to do with the branches
        
        # those are the global branches (jet pt etc)
        # they should be just glued to each other in one vector
        # and zero padded (and mean subtracted and normalised)
        #x_global = MeanNormZeroPad(filename,TupleMeanStd,
        #                           [self.branches[0]],
        #                           [self.branchcutoffs[0]],self.nsamples)
        
        # the second part (the pf candidates) should be treated particle wise
        # an array with (njets, nparticles, nproperties) is created
    
        x_glb  = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                          self.branches[0],
                                          self.branchcutoffs[0],self.nsamples)

        x_pf  = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                         self.branches[1],
                                         self.branchcutoffs[1],self.nsamples)
        
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                         self.branches[2],
                                         self.branchcutoffs[2],self.nsamples)
        
        x_sv = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                        self.branches[3],
                                        self.branchcutoffs[3],self.nsamples)


        # maybe also an image of the energy density of charged particles 
        # should be added
        #x_chmap = createDensityMap(filename,TupleMeanStd,
        #                           'Cpfcan_erel', #use the energy to create the image
        #                           self.nsamples,
        #                           # 7 bins in eta with a total width of 2*0.9
        #                           ['Cpfcan_eta','jet_eta',7,0.9], 
        #                           # 7 bins in phi with a total width of 2*0.9
        #                           ['Cpfcan_phi','jet_phi',7,0.9],
        #                           'nCpfcand',
                                   # the last is an offset because the relative energy as 
                                   # can be found in the ntuples is shifted by 1
        #                           -1)
        
        
        # now, some jets are removed to avoid pt and eta biases
        
        Tuple = self.readTreeFromRootToTuple(filename)
        if self.remove:
            # jets are removed until the shapes in eta and pt are the same as
            # the truth class 'fj_isLight'
            notremoves=weighter.createNotRemoveIndices(Tuple)
            #undef=Tuple[self.undefTruth]
            #notremoves-=undef
        
        if self.weight:
            weights=weighter.getJetWeights(Tuple)
        elif self.remove:
            weights=notremoves
        else:
            print('neither remove nor weight')
            weights=numpy.empty(self.nsamples)
            weights.fill(1.)
            
            
        # create all collections:
        truthtuple =  Tuple[self.truthclasses]
        alltruth=self.reduceTruth(truthtuple)
        
        # remove the entries to get same jet shapes
        if self.remove:
            print('remove')
            weights=weights[notremoves > 0]
            x_glb=x_glb[notremoves > 0]
            x_pf=x_pf[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_sv=x_sv[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
            #x_global=x_global[notremoves > 0]
            #x_chmap=x_chmap[notremoves > 0]        
        
        #newnsamp=x_global.shape[0]
        newnsamp=x_glb.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        # fill everything
        self.w=[weights]
        self.x=[x_pf,x_cpf,x_sv]
        self.y=[alltruth]
        
