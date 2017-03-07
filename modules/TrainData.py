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
        self.branches=[]
        self.branchcutoffs=[]
        
        
    def clear(self):
        import numpy
        self.samplename=''
        self.x=[numpy.array([])]
        self.y=[numpy.array([])]
        self.w=[numpy.array([])]
        
        self.nsamples=0
        
    def getInputShapes(self):
        '''
        returns a list for each input shape. In most cases only one entry
        '''
        outl=[0]
        count=0
        for c in range(len(self.branches)):
            count+=  self.branchcutoffs[c]*len(self.branches[c])
        outl[0]=count
        return outl
        
    def getTruthShapes(self):
        outl=[len(self.getUsedTruth())]
        return outl
        
    def addBranches(self, blist, cutoff=1):
        self.branches.append(blist)
        self.branchcutoffs.append(cutoff)
        
    def getUsedTruth(self):
        if len(self.reducedtruthclasses) > 0:
            return self.reducedtruthclasses
        else:
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
        import h5py
        import numpy
        self.fileTimeOut(fileprefix,120)
        h5f = h5py.File(fileprefix, 'w')
        
        # try "lzf", too, faster, but less compression
        def _writeout(data,idstr,h5F):
            arr=numpy.array(data)
            h5F.create_dataset(idstr+'_shape',data=arr.shape)
            h5F.create_dataset(idstr, data=arr, compression="lzf")
        
        arrw=numpy.array(self.w)
        arrx=numpy.array(self.x)
        arry=numpy.array(self.y)
        
        _writeout(arrw,'w',h5f)
        _writeout(arrx,'x',h5f)
        _writeout(arry,'y',h5f)
        
        arr=numpy.array([self.nsamples],dtype='int')
        h5f.create_dataset('n', data=arr)
        h5f.close()
        return
    
    
        #old pickle implementation
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
        import h5py
        import numpy
        self.fileTimeOut(fileprefix,120)
        h5f = h5py.File(fileprefix,'r')
        
        
        
        import multiprocessing
        import ctypes
        
        def createArr(idstr):
            shapeinfo=numpy.array(h5f[idstr+'_shape'])
            fulldim=1
            for d in shapeinfo:
                fulldim*=d
            # reserve memory for array
            shared_array_base = multiprocessing.Array(ctypes.c_float, int(fulldim))
            shared_array = numpy.ctypeslib.as_array(shared_array_base.get_obj())
            shared_array = shared_array.reshape(shapeinfo)
            return shared_array
            
        
        def _read_arrs(arrw,arrx,arry):
            #still do this sequentially to avoid race in h5py
            h5f['w'].read_direct(arrw)
            h5f['x'].read_direct(arrx)
            h5f['y'].read_direct(arry)
        
        
        def readProcess(arrw,arrx,arry):
            thread=multiprocessing.Process(target=_read_arrs, args=(arrw,arrx,arry))
            thread.start()
            ##now the read process may live on another core
            thread.join()
               
        def arrtolist(Arr):
            out=[]
            for i in range(Arr.shape[0]):
                out.append(Arr[i])
            return out
        
        
        # this might seem weird, but allows to pass by GIL when 
        # each read is performed in a normal python thread
        
        twarr=createArr('w')
        txarr=createArr('x')
        tyarr=createArr('y')
        readProcess(twarr,txarr,tyarr)
        
        
        self.w=arrtolist(numpy.array(twarr))
        self.x=arrtolist(numpy.array(txarr))
        self.y=arrtolist(numpy.array(tyarr))
        
        
        #print(self.x.shape)
        
        self.nsamples=h5f['n']
        self.nsamples=self.nsamples[0]
        h5f.close()
        return
    
    
        #old pickle implementation
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
        
    def readTreeFromRootToTuple(self,filenames):
        '''
        To be used to get the initial tupel for further processing in inherting classes
        Makes sure the number of entries is properly set
        
        can also read a list of files (e.g. to produce weights/removes from larger statistics
        (not fully tested, yet)
        '''
        import ROOT
        from root_numpy import tree2array, root2array
        isalist =  not hasattr(filenames, "split")
        
        if isalist:
            raise Exception('readTreeFromRootToTuple: reading from list does not work, yet')
            for f in filenames:
                self.fileTimeOut(f,120)
            print('add files')
            Tuple = root2array(filenames, treename="deepntuplizer/tree")
            print('done add files')
            return Tuple
        else:    
            self.fileTimeOut(filenames,120) #give eos a minute to recover
            rfile = ROOT.TFile(filenames)
            tree = rfile.Get("deepntuplizer/tree")
            self.nsamples=tree.GetEntries()
            Tuple = tree2array(tree)
            return Tuple
        
        
    def produceMeansFromRootFile(self,filename):
        from preprocessing import meanNormProd
        Tuple=self.readTreeFromRootToTuple(filename)
        return meanNormProd(Tuple)
    
    #overload if necessary
    def produceBinWeighter(self,filename):
        weighter=Weighter() 
        Tuple = self.readTreeFromRootToTuple(filename)
        weight_binXPt = numpy.array([10,25,27.5,30,35,40,45,50,60,75,2000],dtype=float)
        weight_binYEta = numpy.array([0,.4,.8,1.2,1.6,2.,2.4],dtype=float)
        
        weighter.createRemoveProbabilities(Tuple,"jet_pt","jet_eta",[weight_binXPt,weight_binYEta],
                                           classes=self.truthclasses)
       
        weighter.createBinWeights(Tuple,"jet_pt","jet_eta",[weight_binXPt,weight_binYEta],classes=self.truthclasses)
    
        return weighter
        
        
        
        
    def getFlavourClassificationData(self,filename,TupleMeanStd, weighter):
        from stopwatch import stopwatch
        
        sw=stopwatch()
        swall=stopwatch()
        
        import ROOT
        
        self.fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("deepntuplizer/tree")
        self.nsamples=tree.GetEntries()
        
        print('took ', sw.getAndReset(), ' seconds for getting tree entries')
        
        x_all = MeanNormZeroPad(filename,TupleMeanStd,self.branches,self.branchcutoffs,self.nsamples)
        
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        notremoves=weighter.createNotRemoveIndices(Tuple)
        
        print('took ', sw.getAndReset(), ' to create remove indices')
        
        weights=notremoves
        #
        #print('rescale flat branches')
        #x_global_flat = MeanNormApply(Tuple[self.flatbranches],TupleMeanStd)
        #x_global_flat = numpy.array(x_global_flat.tolist())
        #
        #x_all=x_global_flat
        #
        ##def threadFunc(i):
        ##    return MeanNormZeroPad(Tuple[self.deepbranches[i]],TupleMeanStd,self.deepcutoffs[i])
        ##
        ##from multiprocessing.pool import ThreadPool
        ##pool=ThreadPool(len(self.deepbranches))
        ##deepzeropadded=pool.map(threadFunc, range(len(self.deepbranches)))
        ##pool.close()
        ##pool.join()
        #
        #deepzeropadded=[]
        #for i in range(0,len(self.deepbranches)):
        #    deepzeropadded.append(numpy.array([]))
        ##    
        ##
        #print(self.deepbranches)
        #
        #
        #print('rescale deep branches')
        ##deepzeropadded=MeanNormZeroPad(Tuple,TupleMeanStd, self.deepbranches,self.deepcutoffs)
        #for i in range(0,len(self.deepbranches)):
        #    #deepzeropadded[i]=MeanNormZeroPad(Tuple[self.deepbranches[i]],TupleMeanStd,self.deepbranches[i],self.deepcutoffs[i])
        #    deepzeropadded[i]=MeanNormZeroPad(filename,TupleMeanStd,self.deepbranches[i],self.deepcutoffs[i],self.nsamples)
        #    
        #print('concatenate')
        ##simple threading does not help due to pythons weird threading idea
        #for i in range(0,len(self.deepbranches)):
        #    x_all=numpy.concatenate( (x_all, 
        #                              deepzeropadded[i])
        #                             ,axis=1)  
        #
        #
        truthtuple =  Tuple[self.truthclasses]
        alltruth=self.reduceTruth(truthtuple)
        
        print('remove')
        weights=weights[notremoves > 0]
        x_all=x_all[notremoves > 0]
        alltruth=alltruth[notremoves > 0]
       
        newnsamp=x_all.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print('took in total ', swall.getAndReset(),' seconds for conversion')
        
        return weights,x_all,alltruth
        

from preprocessing import MeanNormApply, MeanNormZeroPad
import numpy

class TrainData_Flavour(TrainData):
    '''
    same as TrainData_deepCSV but with 3 truth labels: UDSG C B
    '''


    def __init__(self):
        TrainData.__init__(self)
        
        
    
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        
        weights,x_all,alltruth=self.getFlavourClassificationData(filename,TupleMeanStd, weighter)
        
        self.w=[weights]
        self.x=[x_all]
        self.y=[alltruth]
        
    def reduceTruth(self, tuple_in):
        return numpy.array(tuple_in)
     
        