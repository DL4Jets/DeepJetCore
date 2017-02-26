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
        print('file I/O problems... waiting for filesystem to become available')
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
        pickle.dump(self.w, fd,protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.x, fd,protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.y, fd,protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.nsamples, fd,protocol=pickle.HIGHEST_PROTOCOL)
        fd.close()
        
    def readIn(self,fileprefix):
        import pickle
        import gzip
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
        
        