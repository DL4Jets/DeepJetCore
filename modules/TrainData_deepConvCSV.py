'''
Created on 7 Apr 2017

@author: jkiesele
'''


from TrainData_deepCSV import TrainData_deepCSV

class TrainData_deepConvCSV(TrainData_deepCSV):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        TrainData_deepCSV.__init__(self)
        
        


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
        
        x_global = MeanNormZeroPad(filename,TupleMeanStd,
                                   self.branches[0],
                                   self.branchcutoffs[0],self.nsamples)
        
        
        x_a = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[1],
                                   self.branchcutoffs[1],self.nsamples)
        
        x_b = MeanNormZeroPadParticles(filename,TupleMeanStd,
                                   self.branches[2],
                                   self.branchcutoffs[2],self.nsamples)
        
        
        x_c = MeanNormZeroPadParticles(filename,TupleMeanStd,
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
            x_a=x_a[notremoves > 0]
            x_b=x_b[notremoves > 0]
            x_c=x_c[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
       
        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print(x_global.shape,self.nsamples)

        self.w=[weights]
        self.x=[x_global,x_a,x_b,x_c]
        self.y=[alltruth]