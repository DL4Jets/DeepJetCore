from DeepJetCore.TrainData import TrainData
from DeepJetCore.TrainData import fileTimeOut 
from DeepJetCore import SimpleArray
import numpy


class baseTDTesting(TrainData):
    def __init__(self):
        import numpy
        TrainData.__init__(self)
    
    def readTreeFromRootToTuple(self, filenames, limit=None, branches=None):
        '''
        To be used to get the initial tupel for further processing in inherting classes
        Makes sure the number of entries is properly set
        
        can also read a list of files (e.g. to produce weights/removes from larger statistics
        (not fully tested, yet)
        '''
        
        if branches is None or len(branches) == 0:
            return numpy.array([],dtype='float32')
            
        #print(branches)
        #remove duplicates
        usebranches=list(set(branches))
        tmpbb=[]
        for b in usebranches:
            if len(b):
                tmpbb.append(b)
        usebranches=tmpbb
            
        import ROOT
        from root_numpy import tree2array, root2array
        if isinstance(filenames, list):
            for f in filenames:
                fileTimeOut(f,120)
            print('add files')
            nparray = root2array(
                filenames, 
                treename = self.treename, 
                stop = limit,
                branches = usebranches
                )
            print('done add files')
            return nparray
            print('add files')
        else:    
            fileTimeOut(filenames,120) #give eos a minute to recover
            rfile = ROOT.TFile(filenames)
            tree = rfile.Get(self.treename)
            if not self.nsamples:
                self.nsamples=tree.GetEntries()
            nparray = tree2array(tree, stop=limit, branches=usebranches)
            return nparray
    
    def addBranches(self, br, cut):
        self.branches.append(br)
        self.branchcutoffs.append(cut)

class base_traindata_batchex(baseTDTesting):
    
    def __init__(self):
        import numpy
        TrainData.__init__(self)
        
        #setting DeepJet specific defaults
        self.treename="deepntuplizer/tree"
        self.truthclasses=['class1','class2']

        self.treename="deepntuplizer/tree"
        self.referenceclass='class1'
        
        

        
             
    def reduceTruth(self, tuple_in):
    
        self.reducedtruthclasses=['isB','isBB','isLeptB','isC','isUDS','isG']
        if tuple_in is not None:
            class1 = tuple_in['class1'].view(numpy.ndarray)
        
            class2 = tuple_in['class2'].view(numpy.ndarray)
            
            return numpy.ascontiguousarray(numpy.array(numpy.vstack((class1,class2)).transpose(),dtype='float32'))
  
        
 

class TrainData_testBatch(base_traindata_batchex):
   

    def __init__(self):
        base_traindata_batchex.__init__(self)

    def convertFromSourceFile(self, filename, weighterobjects, istraining):

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
                                   ['x'],
                                   [1],self.nsamples)
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename, branches = ['class1','class2','x'])
        
       
        truthtuple =  Tuple[self.truthclasses]
        
        alltruth=self.reduceTruth(truthtuple)
        
        #print(x_global.shape,x_global[0:10])
        #print(alltruth.shape,alltruth[0:10])
        #print(alltruth.flags)
        
        newnsamp=x_global.shape[0]
        self.nsamples = newnsamp
        
        print(x_global.shape, alltruth.shape, self.nsamples)
        
        truth = SimpleArray(alltruth,name="truth")
        feat = SimpleArray(x_global,name="features0")

        return [feat], [truth], []
        
        
        
        
        
        

class TrainDataDeepJet_base(baseTDTesting):
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
        
        

        
        self.branches=[]
        self.reduceTruth(None)
    
    def reduceTruth(self, tuple_in):
        pass
    

    



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
            
            return numpy.ascontiguousarray(numpy.array(numpy.vstack((b,bb+gbb,lepb,c+cc+gcc,uds,g)).transpose() ,dtype='float32'))


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

        
        
    def convertFromSourceFile(self, filename, weighterobjects, istraining):
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
        
        x_cpf = MeanNormZeroPadParticles(filename,None,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        x_npf = MeanNormZeroPadParticles(filename,None,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        
        x_sv = MeanNormZeroPadParticles(filename,None,
                                   self.branches[3],
                                   self.branchcutoffs[3],self.nsamples)
        
        
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        
        
        truthtuple =  Tuple[self.truthclasses]
        #print(self.truthclasses)
        alltruth=self.reduceTruth(truthtuple)
        
       
        print(x_global.shape,self.nsamples)

        return [x_global,x_cpf,x_npf,x_sv], [alltruth], []
        
        
