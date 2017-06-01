
from TrainData import TrainData,fileTimeOut



class TrainData_FatJet(TrainData):
    
    def __init__(self):
        '''
        This class is meant as a base class for the FatJet studies
        You will not need to edit it for trying out things
        '''
        TrainData.__init__(self)
        
        #define truth:
        self.undefTruth=['isUndefined']
        self.truthclasses=['isQCD','isTop','isW','isUnmatched']
        self.referenceclass='isQCD'
        
        #this is only needed because the truth definitions are different from deepFlavour
        self.allbranchestoberead=[]
        self.registerBranches(self.undefTruth)
        self.registerBranches(self.truthclasses)
        
        
    
    def reduceTruth(self, tuple_in):
        import numpy
        self.reducedtruthclasses=['isQCD','isTop','isW']
        if tuple_in is not None:
            q = tuple_in['isQCD'].view(numpy.ndarray)
            w = tuple_in['isTop'].view(numpy.ndarray)
            t = tuple_in['isW'].view(numpy.ndarray)
            
            return numpy.vstack((q,w,t)).transpose()  
        
        
        
#######################################
        
        
class TrainData_FatJet_Test(TrainData_FatJet):
    
    def __init__(self):
        '''
        This is an example data format description for FatJet studies
        '''
        TrainData_FatJet.__init__(self)
        
        #example of how to register global branches
        self.addBranches(['jet_pt', 'jet_eta','nCpfcand','nNpfcand','nsv','npv'])

        #example of pf candidate branches
        #up to 60 pf candidates will be stored in the data format
        self.addBranches(['Cpfcan_relpt',
                          'Cpfcan_releta'
                              ],
                             60) 
        
        #branches that are used directly in the following function 'readFromRootFile'
        #this is a technical trick to speed up the conversion
        self.registerBranches(['Cpfcan_erel','Cpfcan_eta','Cpfcan_phi',
                               'Npfcan_erel','Npfcan_eta','Npfcan_phi',
                               'nCpfcand','nNpfcand',
                               'jet_eta','jet_phi'])
        
        
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
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   [self.branches[0]],
                                   [self.branchcutoffs[0]],self.nsamples)
        
        # the second part (the pf candidates) should be treated particle wise
        # an array with (njets, nparticles, nproperties) is created
        x_cpf = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        # maybe also an image of the energy density of charged particles 
        # should be added
        x_chmap = createDensityMap(filename,TupleMeanStd,
                                   'Cpfcan_erel', #use the energy to create the image
                                   self.nsamples,
                                   # 7 bins in eta with a total width of 2*0.9
                                   ['Cpfcan_eta','jet_eta',7,0.9], 
                                   # 7 bins in phi with a total width of 2*0.9
                                   ['Cpfcan_phi','jet_phi',7,0.9],
                                   'nCpfcand',
                                   # the last is an offset because the relative energy as 
                                   # can be found in the ntuples is shifted by 1
                                   -1)
        
        
        # now, some jets are removed to avoid pt and eta biases
        
        Tuple = self.readTreeFromRootToTuple(filename)
        if self.remove:
            # jets are removed until the shapes in eta and pt are the same as
            # the truth class 'isQCD'
            notremoves=weighter.createNotRemoveIndices(Tuple)
            undef=Tuple[self.undefTruth]
            notremoves-=undef
        
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
            x_global=x_global[notremoves > 0]
            x_cpf=x_cpf[notremoves > 0]
            x_chmap=x_chmap[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
        
        
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        # fill everything
        self.w=[weights]
        self.x=[x_global,x_cpf,x_chmap]
        self.y=[alltruth]
        