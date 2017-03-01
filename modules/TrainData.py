'''
Created on 20 Feb 2017

@author: jkiesele
'''


from Weighter import Weighter

class TrainData(object):
    '''
    Base class for batch-wise training of the DNN
    '''
    def __init__(self):
        '''
        Constructor
        '''
        self.clear()
        
        self.truthclasses=[]
        self.reducedtruthclasses=[]
        self.regressionclasses=[]
        
        self.flatbranches=[]
        self.deepbranches=[]
        self.deepcutoffs=[]
        
    def clear(self):

        self.samplename=''
        self.x=[[]]
        self.y=[[]]
        self.w=[[]]
        
        self.nsamples=0
        
        
    def addDeepBranches(self, blist, cutoff):
        self.deepbranches.append(blist)
        self.deepcutoffs.append(cutoff)
        
    def getUsedTruth(self):
        if len(self.reducedtruthclasses) > 0:
            return self.reducedtruthclasses
        return self.truthclasses
    
    def addFromRootFile(self,fileName):
        '''
        Adds from a root file and randomly shuffles the input
        '''
        raise Exception('to be implemented')
        #just call read from root (virtual in python??), and mix with existing x,y,weight


    def __reduceTruth(self,tuple_in):
        import numpy
        return numpy.array(tuple_in.tolist())

    def fileTimeOut(self,fileName, timeOut):
        '''
        simple wait function in case the file system has a glitch.
        waits until the dir, the file should be stored in/read from, is accessible
        again, or the the timeout
        '''
        import os
        filepath=os.path.dirname(fileName)
        if len(filepath) < 1:
            filepath = '.'
        if os.path.isdir(filepath):
            return
        import time
        counter=0
        print('file I/O problems... waiting for filesystem to become available for '+fileName)
        while not os.path.isdir(filepath):
            if counter > timeOut:
                print('...file could not be opened within '+str(timeOut)+ ' seconds')
            counter+=1
            time.sleep(1)


    def writeOut(self,fileprefix):
        import pickle
        import gzip
        self.fileTimeOut(fileprefix,120) #give eos a minute to recover
        fd=gzip.open(fileprefix,'wb')
        pprot=2 #compatibility to python 2
        pickle.dump(self.w, fd,protocol=pprot)
        pickle.dump(self.x, fd,protocol=pprot)
        pickle.dump(self.y, fd,protocol=pprot)
        pickle.dump(self.nsamples, fd,protocol=pprot)
        fd.close()
        
    def readIn(self,fileprefix):
        import pickle
        import gzip
        self.samplename=fileprefix
        self.fileTimeOut(fileprefix,120) #give eos a minute to recover
        fd=gzip.open(fileprefix,'rb')
        self.w=pickle.load(fd)
        self.x=pickle.load(fd)
        self.y=pickle.load(fd)
        self.nsamples=pickle.load(fd)
        fd.close()
        
    def readTreeFromRootToTuple(self,filename):
        '''
        To be used to get the initial tupel for further processing in inherting classes
        Makes sure the number of entries is properly set
        '''
        import ROOT
        from root_numpy import tree2array
        self.fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples=tree.GetEntries()
        Tuple = tree2array(tree)
        return Tuple
        
        
    def produceMeansFromRootFile(self,filename):
        from preprocessing import meanNormProd
        Tuple=self.readTreeFromRootToTuple(filename)
        return meanNormProd(Tuple)
    
    def produceBinWeighter(self,filename):
        return Weighter() #overload in derived classes
        
        
        
        
        
        
        

from Weighter import Weighter
from preprocessing import MeanNormApply, MeanNormZeroPad
import numpy

class TrainData_Flavour(TrainData):
    '''
    same as TrainData_deepCSV but with 3 truth labels: UDSG C B
    '''


    def __init__(self):
        TrainData.__init__(self)
        
        
    def produceBinWeighter(self,filename):
        weighter=Weighter() 
        Tuple = self.readTreeFromRootToTuple(filename)
        weight_binXPt = numpy.array([10,25,30,35,40,45,50,60,75,2000],dtype=float)
        weight_binYEta = numpy.array([0,.4,.8,1.2,1.6,2.,2.4],dtype=float)
        weighter.createBinWeights(Tuple,"jet_pt","jet_eta",[weight_binXPt,weight_binYEta],classes=self.truthclasses)
        print('weights produced')
        return weighter
    
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        
        Tuple = self.readTreeFromRootToTuple(filename)
        weights=weighter.getJetWeights(Tuple)
        
        
        
        x_global_flat = MeanNormApply(Tuple[self.flatbranches],TupleMeanStd)
        x_global_flat = numpy.array(x_global_flat.tolist())
        
        x_all=x_global_flat
        
        for i in range(0,len(self.deepbranches)):
            x_all=numpy.concatenate( (x_all, 
                                      MeanNormZeroPad(Tuple[self.deepbranches[i]],TupleMeanStd, self.deepcutoffs[i])
                                     ),axis=1)  
        
       
        truthtuple =  Tuple[self.truthclasses]
        alltruth=self.reduceTruth(truthtuple)
        
        #####needs to be filled in any implementation
        
        self.w=[weights]
        self.x=[x_all]
        self.y=[alltruth]
        
     
        