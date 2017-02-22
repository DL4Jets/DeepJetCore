'''
Created on 20 Feb 2017

@author: jkiesele
'''




class TrainData(object):
    '''
    Base class for batch-wise training of the DNN
    '''
    def __init__(self):
        '''
        Constructor
        '''
        
        self.x=[[]]
        self.y=[[]]
        self.w=[[]]
        
        self.nsamples=0
        
    def clear(self):

        self.x=[[]]
        self.y=[[]]
        self.w=[[]]
        
        self.nsamples=0
        
    def addFromRootFile(self,fileName):
        '''
        Adds from a root file and randomly shuffles the input
        '''
        raise Exception('to be implemented')
        #just call read from root (virtual in python??), and mix with existing x,y,weight

    def writeOut(self,fileprefix):
        import pickle
        import gzip
        fd=gzip.open(fileprefix,'wb')
        pickle.dump(self.w, fd,protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.x, fd,protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.y, fd,protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.nsamples, fd,protocol=pickle.HIGHEST_PROTOCOL)
        fd.close()
        
    def readIn(self,fileprefix):
        import pickle
        import gzip
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
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples=tree.GetEntries()
        Tuple = tree2array(tree)
        return Tuple
        
        