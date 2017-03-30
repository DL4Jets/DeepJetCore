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
        
        self.readthread=None
        self.readdone=None
        
        self.remove=True    
        self.weight=False

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
        outl=[]
        for x in self.x:
            outl.append(x.shape)
        shapes=[]
        for s in outl:
            _sl=[]
            for i in range(len(s)):
                if i:
                    _sl.append(s[i])
            s=(_sl)
            shapes.append(s)

        return shapes
        
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
        def _writeoutListinfo(arrlist,fidstr,h5F):
            arr=numpy.array([len(arrlist)])
            h5F.create_dataset(fidstr+'_listlength',data=arr)
            for i in range(len(arrlist)):
                idstr=fidstr+str(i)
                h5F.create_dataset(idstr+'_shape',data=arrlist[i].shape)
            
        def _writeoutArrays(arrlist,fidstr,h5F):    
            for i in range(len(arrlist)):
                idstr=fidstr+str(i)
                arr=arrlist[i]
                h5F.create_dataset(idstr, data=arr, compression="lzf")
        
        
        arr=numpy.array([self.nsamples],dtype='int')
        h5f.create_dataset('n', data=arr)

        _writeoutListinfo(self.w,'w',h5f)
        _writeoutListinfo(self.x,'x',h5f)
        _writeoutListinfo(self.y,'y',h5f)

        _writeoutArrays(self.w,'w',h5f)
        _writeoutArrays(self.x,'x',h5f)
        _writeoutArrays(self.y,'y',h5f)
        
        h5f.close()
       
       
    def __createArr(self,shapeinfo):
        import ctypes
        import multiprocessing
        fulldim=1
        for d in shapeinfo:
            fulldim*=d 
        if fulldim < 0: #catch some weird things that happen when there is a file IO error
            fulldim=0 
        # reserve memory for array
        shared_array_base = multiprocessing.RawArray(ctypes.c_float, int(fulldim))
        shared_array = numpy.ctypeslib.as_array(shared_array_base)#.get_obj())
        #print('giving shape',shapeinfo)
        shared_array = shared_array.reshape(shapeinfo)
        #print('gave shape',shapeinfo)
        return shared_array
       
    def readIn_async(self,fileprefix):
        
        if self.readthread:
            print('\nTrainData::readIn_async: started new read before old was finished. Intended? Waiting for first to finish...\n')
            self.readIn_join()
            
        #print('read')
        import h5py
        import numpy
        import multiprocessing
        
        #print('\ninit async read\n')
        
        self.fileTimeOut(fileprefix,120)
        #print('\nfile access ok\n')
        self.samplename=fileprefix
        
        self.h5f = h5py.File(fileprefix,'r')
        
        def _readListInfo(idstr):
            sharedlist=[]
            shapeinfos=[]
            wlistlength=self.h5f[idstr+'_listlength'][0]
            #print(idstr,'list length',wlistlength)
            for i in range(wlistlength):
                sharedlist.append(numpy.array([]))
                iidstr=idstr+str(i)
                shapeinfo=numpy.array(self.h5f[iidstr+'_shape'])
                #print('read shape info',shapeinfo)
                shapeinfos.append(shapeinfo)
            return sharedlist, shapeinfos
        
        def _read_arrs(arrwl,arrxl,arryl,doneVal):
            idstrs=['w','x','y']
            alllists=[arrwl,arrxl,arryl]
            for j in range(len(idstrs)):
                fidstr=idstrs[j]
                arl=alllists[j]
                for i in range(len(arl)):
                    idstr=fidstr+str(i)
                    #print('reading',idstr)
                    self.h5f[idstr].read_direct(arl[i])
                    #print('done reading',idstr)
            self.readdone.value=True
        
        self.nsamples=self.h5f['n']
        self.nsamples=self.nsamples[0]
        
        self.w_list,w_shapes=_readListInfo('w')
        self.x_list,x_shapes=_readListInfo('x')
        self.y_list,y_shapes=_readListInfo('y')
        
        for i in range(len(self.w_list)):
            self.w_list[i]=self.__createArr(w_shapes[i])
            
        for i in range(len(self.x_list)):
            self.x_list[i]=self.__createArr(x_shapes[i])
            
        for i in range(len(self.y_list)):
            self.y_list[i]=self.__createArr(y_shapes[i])

        self.readdone=multiprocessing.Value('b',False)
        self.readthread=multiprocessing.Process(target=_read_arrs, args=(self.w_list,self.x_list,self.y_list,self.readdone))
        self.readthread.start()
        
        #print('\nstarted async thread\n')
     
    def readIn_join(self):
        #print('joining async read')
        while not self.readdone.value: 
            #needs to be done - it can come to deadlocks because of wrong locking in python..
            #use the shared self.readthread as soft lock
            self.readthread.join(1)
        #print('joined async read')
        
        #import copy
        self.w=(self.w_list)#get all arrays back form the shared memory
        self.x=(self.x_list)
        self.y=(self.y_list)
        self.h5f.close()
        self.w_list=None
        self.x_list=None
        self.y_list=None
        self.readthread.terminate()
        self.readthread=None
        self.readdone=None
        
    def readIn(self,fileprefix):
        
        self.readIn_async(fileprefix)
        self.readIn_join()
        
        
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
        
        if self.remove:
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
    
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        
        x_all = MeanNormZeroPad(filename,TupleMeanStd,self.branches,self.branchcutoffs,self.nsamples)
        
        print('took ', sw.getAndReset(), ' seconds for mean norm and zero padding (C module)')
        
        notremoves=numpy.array([])
        weights=numpy.array([])
        if self.remove:
            notremoves=weighter.createNotRemoveIndices(Tuple)
            weights=notremoves
            print('took ', sw.getAndReset(), ' to create remove indices')
        elif self.weight:
            print('creating weights')
            weights= weighter.getJetWeights(Tuple)
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
            x_all=x_all[notremoves > 0]
            alltruth=alltruth[notremoves > 0]
       
        newnsamp=x_all.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        self.nsamples = newnsamp
        
        print('took in total ', swall.getAndReset(),' seconds for conversion')
        
        return weights,x_all,alltruth, notremoves
        

from preprocessing import MeanNormApply, MeanNormZeroPad
import numpy

class TrainData_Flavour(TrainData):
    '''
    same as TrainData_deepCSV but with 3 truth labels: UDSG C B
    '''


    def __init__(self):
        TrainData.__init__(self)
        self.clear()
        
    
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        
        weights,x_all,alltruth, _ =self.getFlavourClassificationData(filename,TupleMeanStd, weighter)
        
        self.w=[weights]
        self.x=[x_all]
        self.y=[alltruth]
        
     
        
