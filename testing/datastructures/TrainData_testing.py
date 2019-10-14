from DeepJetCore.TrainData import TrainData
from DeepJetCore.TrainData import fileTimeOut 
import numpy


class base_traindata_batchex(TrainData):
    
    def __init__(self):
        import numpy
        TrainData.__init__(self)
        
        #setting DeepJet specific defaults
        self.treename="deepntuplizer/tree"
        self.truthclasses=['class1','class2']

        self.treename="deepntuplizer/tree"
        self.referenceclass='class1'
        
        
        self.registerBranches(self.truthclasses)
        self.registerBranches(['x'])
        
        self.weightbranchX='x'
        self.weightbranchY='x'
        
        self.weight_binX = numpy.array([-1,0.9,2.0],dtype=float)
        
        self.weight_binY = numpy.array(
            [-1,0.9,2.0],
            dtype=float
            )

        
             
        def reduceTruth(self, tuple_in):
        
            self.reducedtruthclasses=['isB','isBB','isLeptB','isC','isUDS','isG']
            if tuple_in is not None:
                class1 = tuple_in['class1'].view(numpy.ndarray)
            
                class2 = tuple_in['class2'].view(numpy.ndarray)
                
                return numpy.vstack((class1,class2)).transpose()    
  
        
 

class TrainData_testBatch(base_traindata_batchex):
   

    def __init__(self):
        base_traindata_batchex.__init__(self)
        self.addBranches(['x'])

    def readFromRootFile(self,filename,TupleMeanStd,weighter):


        from DeepJetCore.preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles
        import numpy
        from DeepJetCore.stopwatch import stopwatch
        
        sw=stopwatch()
        swall=stopwatch()
        
        import ROOT
        
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples=tree.GetEntries()
        
        print('took ', sw.getAndReset(), ' seconds for getting tree entries')
        
        
        # split for convolutional network
        
        x_global = MeanNormZeroPad(filename,None,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
       
        truthtuple =  Tuple[self.truthclasses]
        alltruth=self.reduceTruth(truthtuple)
        
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[]
        self.x=[x_global]
        self.y=[alltruth]
        
        
        
        
        

class TrainDataDeepJet_base(TrainData):
    '''
    Base class for DeepJet.
    TO create own b-tagging trainings, please inherit from this class
    '''
    
    def __init__(self):
        import numpy
        TrainData.__init__(self)
        
        #setting DeepJet specific defaults
        self.treename="deepntuplizer/tree"
        self.undefTruth=['isUndefined']
        self.referenceclass='isB'
        self.truthclasses=['isB','isBB','isGBB','isLeptonicB','isLeptonicB_C','isC','isCC',
                           'isGCC','isUD','isS','isG','isUndefined']
        
        
        #standard branches
        self.registerBranches(self.undefTruth)
        self.registerBranches(self.truthclasses)
        self.registerBranches(['jet_pt','jet_eta'])
        
        self.weightbranchX='jet_pt'
        self.weightbranchY='jet_eta'
        
        self.weight_binX = numpy.array([
                10,25,30,35,40,45,50,60,75,100,
                125,150,175,200,250,300,400,500,
                600,2000],dtype=float)
        
        self.weight_binY = numpy.array(
            [-2.5,-2.,-1.5,-1.,-0.5,0.5,1,1.5,2.,2.5],
            dtype=float
            )
        
        
             
        self.reduceTruth(None)
        

class TrainData_fullTruth_base(TrainDataDeepJet_base):
    def __init__(self):
        TrainDataDeepJet_base.__init__(self)
        self.clear()
        
    def reduceTruth(self, tuple_in):
        
        self.reducedtruthclasses=['isB','isBB','isLeptB','isC','isUDS','isG']
        if tuple_in is not None:
            b = tuple_in['isB'].view(numpy.ndarray)
            
            bb = tuple_in['isBB'].view(numpy.ndarray)
            gbb = tuple_in['isGBB'].view(numpy.ndarray)
            
            
            bl = tuple_in['isLeptonicB'].view(numpy.ndarray)
            blc = tuple_in['isLeptonicB_C'].view(numpy.ndarray)
            lepb=bl+blc
           
            c = tuple_in['isC'].view(numpy.ndarray)
            cc = tuple_in['isCC'].view(numpy.ndarray)
            gcc = tuple_in['isGCC'].view(numpy.ndarray)
           
            ud = tuple_in['isUD'].view(numpy.ndarray)
            s = tuple_in['isS'].view(numpy.ndarray)
            uds=ud+s
            
            g = tuple_in['isG'].view(numpy.ndarray)
            
            
            return numpy.vstack((b,bb+gbb,lepb,c+cc+gcc,uds,g)).transpose() 


class TrainData_testQueue(TrainData_fullTruth_base):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        TrainData_fullTruth_base.__init__(self)
        
        
        self.addBranches(['jet_pt', 'jet_eta',
                          'nCpfcand','nNpfcand',
                          'nsv','npv',
                          'TagVarCSV_trackSumJetEtRatio', 
                          'TagVarCSV_trackSumJetDeltaR', 
                          'TagVarCSV_vertexCategory', 
                          'TagVarCSV_trackSip2dValAboveCharm', 
                          'TagVarCSV_trackSip2dSigAboveCharm', 
                          'TagVarCSV_trackSip3dValAboveCharm', 
                          'TagVarCSV_trackSip3dSigAboveCharm', 
                          'TagVarCSV_jetNSelectedTracks', 
                          'TagVarCSV_jetNTracksEtaRel'])
       
        self.addBranches(['Cpfcan_BtagPf_trackEtaRel',
                          'Cpfcan_BtagPf_trackPtRel',
                          'Cpfcan_BtagPf_trackPPar',
                          'Cpfcan_BtagPf_trackDeltaR',
                          'Cpfcan_BtagPf_trackPParRatio',
                          'Cpfcan_BtagPf_trackSip2dVal',
                          'Cpfcan_BtagPf_trackSip2dSig',
                          'Cpfcan_BtagPf_trackSip3dVal',
                          'Cpfcan_BtagPf_trackSip3dSig',
                          'Cpfcan_BtagPf_trackJetDistVal',
                          #'Cpfcan_BtagPf_trackJetDistSig',
                          
                          'Cpfcan_ptrel', 
                          'Cpfcan_drminsv',
                          #'Cpfcan_fromPV',
                          'Cpfcan_VTX_ass',
                          'Cpfcan_puppiw',
                          'Cpfcan_chi2',
                          'Cpfcan_quality'
                              ],
                             25)
        
        
        self.addBranches(['Npfcan_ptrel',
                          'Npfcan_deltaR',
                          'Npfcan_isGamma',
                          'Npfcan_HadFrac',
                          'Npfcan_drminsv',
                          'Npfcan_puppiw'
                          ],
                         25)
        
        
        self.addBranches(['sv_pt',
                          'sv_deltaR',
                          'sv_mass',
                          'sv_ntracks',
                          'sv_chi2',
                          'sv_normchi2',
                          'sv_dxy',
                          'sv_dxysig',
                          'sv_d3d',
                          'sv_d3dsig',
                          'sv_costhetasvpv',
                          'sv_enratio',
                          ],
                          4)

        
        
       
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        from DeepJetCore.preprocessing import MeanNormApply, MeanNormZeroPad, MeanNormZeroPadParticles
        import numpy
        from DeepJetCore.stopwatch import stopwatch
        
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
