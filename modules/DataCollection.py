'''
Created on 21 Feb 2017

@author: jkiesele
'''
#from tensorflow.contrib.labeled_tensor import batch
#from builtins import list
from Weighter import Weighter
from TrainData import TrainData
#for convenience


class DataCollection(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.clear()
        
        
    def clear(self):
        self.samples=[]
        self.dataDir=""
        self.sampleentries=[]
        self.originRoots=[]
        self.nsamples=0
        self.useweights=True
        self.__batchsize=1
        self.filesPreRead=2
        self.isTrain=True
        #self.dataclass=TrainData #for future implementations
        
    def removeLast(self):
        self.samples.pop()
        self.nsamples-=self.sampleentries[-1]
        self.sampleentries.pop()
        self.originRoots.pop()
        
        
    def setBatchSize(self,bsize):
        if bsize > self.nsamples:
            raise Exception('Batch size must not be bigger than total sample size')
        self.__batchsize=bsize
        
    def getSamplesPerEpoch(self):
        #modify by batch split
        count=self.getNBatchesPerEpoch()
        if count != 1:
            return count*self.__batchsize #final
        else:
            return self.nsamples
        
    
    def getNBatchesPerEpoch(self):
        if self.__batchsize <= 1:
            return 1
        count=0
        while (count+1)*self.__batchsize <= self.nsamples:
            count+=1
        return count
        
    def writeToFile(self,filename):
        import pickle
        fd=open(filename,'wb')
        pickle.dump(self.samples, fd,protocol=0 )
        pickle.dump(self.sampleentries, fd,protocol=0 )
        pickle.dump(self.originRoots, fd,protocol=0 )
        pickle.dump(self.nsamples, fd,protocol=0 )
        pickle.dump(self.useweights, fd,protocol=0 )
        pickle.dump(self.__batchsize, fd,protocol=0 )
        fd.close()
        
    def readFromFile(self,filename):
        import pickle
        fd=open(filename,'rb')
        self.samples=pickle.load(fd)
        self.sampleentries=pickle.load(fd)
        self.originRoots=pickle.load(fd)
        self.nsamples=pickle.load(fd)
        self.useweights=pickle.load(fd)
        self.__batchsize=pickle.load(fd)
        fd.close()
        import os
        self.dataDir=os.path.dirname(os.path.abspath(filename))
        self.dataDir+='/'
        #check if files exist
        for f in self.originRoots:
            if not f.endswith(".root"): continue
            if not os.path.isfile(f):
                print('not found: '+f)
                raise Exception('original root file not found')
        for f in self.samples:
            fpath=self.getSamplePath(f)
            if not os.path.isfile(fpath):
                print('not found: '+fpath)
                raise Exception('sample file not found')
        
        
    def readRootListFromFile(self,file):
        import os
        self.clear()
        
        fdir=os.path.dirname(file)
        fdir=os.path.abspath(fdir)
        lines = [line.rstrip('\n') for line in open(file)]
        for line in lines:
            if len(line) < 1: continue
            self.originRoots.append(fdir+'/'+line)
    
        if len(self.originRoots)<1:
            raise Exception('root samples list empty')
        
        
    def split(self,ratio):
        '''
        ratio is self/(out+self)
        returns out
        modifies itself
        '''
        
        itself=DataCollection()
        out=DataCollection()
        
        nsamples=len(self.samples)
    
        for i in range(0, nsamples):
            if i/nsamples < ratio and i < nsamples-1:
                itself.samples.append(self.samples[i])
                itself.sampleentries.append(self.sampleentries[i])
                itself.originRoots.append(self.originRoots[i])
                itself.nsamples+=self.sampleentries[i]
            else:
                out.samples.append(self.samples[i])
                out.sampleentries.append(self.sampleentries[i])
                out.originRoots.append(self.originRoots[i])
                out.nsamples+=self.sampleentries[i]
           
        
        out.useweights=self.useweights
        
        
        itself.setBatchSize(self.__batchsize)
        out.dataDir=self.dataDir
        out.setBatchSize(self.__batchsize)
        
        self.samples=itself.samples
        self.sampleentries=itself.sampleentries
        self.originRoots=itself.originRoots
        self.nsamples=itself.nsamples
        #self.useweights=True
        self.setBatchSize(self.__batchsize) #check if still ok
        
        return out
    
    
    def recoverCreateDataFromRootFromSnapshot(self, snapshotfile, dataclass):
        import os
        snapshotfile=os.path.abspath(snapshotfile)
        self.readFromFile(snapshotfile)
        td=dataclass
        if len(self.originRoots) < 1:
            return
        means=td.produceMeansFromRootFile(self.originRoots[0])
        weighter=td.produceBinWeighter(self.originRoots[0])
        outputDir=os.path.dirname(snapshotfile)+'/'
        self.dataDir=outputDir
        finishedsamples=len(self.samples)
        for i in range(finishedsamples, len(self.originRoots)):
            if not self.originRoots[i].endswith('.root'): continue
            print ('creating '+ str(type(dataclass)) +' data from '+self.originRoots[i])
            self.__writeData(self.originRoots[i], means, weighter, outputDir, td)
            
        self.writeToFile(outputDir+'/dataCollection.dc')
    
    def createDataFromRoot(self,dataclass, outputDir):
        '''
        Also creates a file list of the output files
        After the operation, the object will point to the already processed
        files (not root files)
        Writes out a snapshot of itself after every successfully written output file
        to recover the data until a possible error occurred
        '''
        import os
        import numpy
        outputDir+='/'
        if os.path.isdir(outputDir):
            raise Exception('output dir must not exist')
        os.mkdir(outputDir)
        self.dataDir=outputDir
        self.nsamples=0
        self.samples=[]
        self.sampleentries=[]
        means=numpy.array([])
        weighter=Weighter()
        firstSample=True
        for sample in self.originRoots:
            td=dataclass
            print ('creating '+ str(type(dataclass)) +' data from '+sample)
            os.path.abspath(sample)
            
            if firstSample:
                print('producing means')
                means=td.produceMeansFromRootFile(sample)
                print('producing bin weights')
                weighter=td.produceBinWeighter(sample)
                firstSample=False
                
            self.__writeData(sample, means, weighter,outputDir, td)
            
        
    def __writeData(self,sample,means, weighter,outputDir,td):
        import os
        td.readFromRootFile(sample,means, weighter) 
        newname=os.path.basename(sample).rsplit('.', 1)[0]
        newpath=os.path.abspath(outputDir+newname+'.z')
        print('writing '+newname+'.z')
        td.writeOut(newpath)
        self.samples.append(newname+'.z')
        self.nsamples+=td.nsamples
        self.sampleentries.append(td.nsamples)
        td.clear()
        self.writeToFile(outputDir+'/snapshot.dc')
        
    def convertListOfRootFiles(self, inputfile, dataclass, outputDir):
        self.readRootListFromFile(inputfile)
        self.createDataFromRoot(dataclass, outputDir)
        self.writeToFile(outputDir+'/dataCollection.dc')
        
    def getAllLabels(self, dataclass):
        return self.__stackData(dataclass,'y')
    
    def getAllFeatures(self, dataclass):
        return self.__stackData(dataclass,'x')
        
    def getAllWeights(self, dataclass):
        return self.__stackData(dataclass,'w')
    
    
    def getSamplePath(self,samplefile):
        #for backward compatibility
        if samplefile[0] == '/':
            return samplefile
        return self.dataDir+'/'+samplefile
    
    def __stackData(self, dataclass, selector):
        import numpy
        td=dataclass
        out=[]
        firstcall=True
        for sample in self.samples:
            td.readIn(self.getSamplePath(sample))
            #make this generic
            thislist=[]
            if selector == 'x':
                thislist=td.x
            if selector == 'y':
                thislist=td.y
            if selector == 'w':
                thislist=td.w
               
            if firstcall:
                out=thislist
                firstcall=False
            else:
                for i in range(0,len(thislist)):
                    if selector == 'w':
                        out[i] = numpy.append(out[i],thislist[i])
                    else:
                        out[i] = numpy.vstack((out[i],thislist[i]))
                
        return out
    
        
    
        
    def generator(self, dataclass):
        # the output of one call defines the batch size
        # we can use this! -> count the number of calls internally, and realise changing batch sizes
        # 
        # but maybe this means, we need to read in two files and then merge the overlap,
        # maybe throw awy parts of it or similar
        #
        #
        # save the read-in arrays in a list and pass parts of it as batches to keras
        # for the overlap regions, merge, depending on the batch size and the sample size and
        # ... try to throw aways the rest as soon as possible
        import numpy
        td=dataclass

        
        totalbatches=self.getNBatchesPerEpoch()
        processedbatches=0
        
        xstored=[numpy.array([])]
        dimx=0
        ystored=[]
        dimy=0
        wstored=[]
        dimw=0
        nextfiletoread=0
        
        xout=[]
        yout=[]
        wout=[]
        
        #read the next sample in advance
        from threading import Thread
        def readTDThread(ttd,samplefile):
            ttd.readIn(samplefile)
        
        readsample=self.getSamplePath(self.samples[0])
        #readthread=Thread(target=readTDThread, args=(td,readsample)) #read first directly
        #readthread.start()
        
        while 1:
            
            if processedbatches == totalbatches:
                processedbatches=0
            
            if processedbatches == 0: #reset buffer and start new
                nextfiletoread=0
                xstored=[numpy.array([])]
                dimx=0
                ystored=[]
                dimy=0
                wstored=[]
                dimw=0
                
            
            batchcomplete=False
            doSplit=True
            
            lastbatchrest=xstored[0].shape[0]
            if lastbatchrest >= self.__batchsize:
                batchcomplete = True
                
            while not batchcomplete:
               
                if nextfiletoread ==0:
                    readthread=Thread(target=readTDThread, args=(td,readsample)) #read first directly
                    readthread.start()
                
                readthread.join()
                #td.readIn(readsample)
                #get the format right in the first read
                if nextfiletoread == 0 or xstored[0].shape[0] ==0: #either first file or nothing left, so no need to append
                    
                    xstored=td.x
                    
                    dimx=len(xstored)
                    ystored=td.y
                    dimy=len(ystored)
                    wstored=td.w
                    dimw=len(wstored)
                    if not self.useweights:
                        dimw=0
                    xout=[]
                    yout=[]
                    wout=[]
                    for i in range(0,dimx):
                        xout.append([])
                        yout.append([])
                    for i in range(0,dimw):
                        wout.append([])
                else:
                    
                    #this can be done more efficiently if it is assumed that samplesize is always larger than 
                    #batchsize... then tdx could be split before stacking it (much faster)
                    #if batchsize < td.nsamples ... then it can be done dynamically
                    
                    #this will be true for almost all situations and increases performance
                    if self.__batchsize < td.nsamples:
                        for i in range(0,dimx):
                            splitted=numpy.split(td.x[i],[self.__batchsize-lastbatchrest])
                            xout[i] = numpy.vstack((xstored[i],splitted[0]))
                            xstored[i]= splitted[1]
                        
                        for i in range(0,dimy):
                            splitted=numpy.split(td.y[i],[self.__batchsize-lastbatchrest])
                            yout[i] = numpy.vstack((ystored[i],splitted[0]))
                            ystored[i]= splitted[1]
                        
                        for i in range(0,dimw):
                            splitted=numpy.split(td.w[i],[self.__batchsize-lastbatchrest])
                            wout[i] = numpy.append(wstored[i],splitted[0])
                            wstored[i]= splitted[1]
                            
                            
                        doSplit = False
                        batchcomplete = True
                    
                    else:  ##this alternative way is more generic but requires more performance
                        for i in range(0,dimx):
                            xstored[i]=numpy.vstack((xstored[i],td.x[i]))
                        
                        for i in range(0,dimy):
                            ystored[i]=numpy.vstack((ystored[i],td.y[i]))
                        
                        for i in range(0,dimw):
                            wstored[i]=numpy.append(wstored[i],td.w[i])
                    
                if xstored[0].shape[0] >= self.__batchsize:
                    batchcomplete = True
                     
                td.clear()
                nextfiletoread+=1
                if nextfiletoread >= len(self.samples):
                    nextfiletoread=0
                
                readsample=self.getSamplePath(self.samples[nextfiletoread])
                if nextfiletoread > 0:
                    readthread=Thread(target=readTDThread, args=(td,readsample)) #read first directly
                    readthread.start()
                
            if batchcomplete and doSplit:
                for i in range(0,dimx):
                    splitted=numpy.split(xstored[i],[self.__batchsize])
                    xstored[i] = splitted[1]
                    xout[i] = splitted[0]
                for i in range(0,dimy):
                    splitted=numpy.split(ystored[i],[self.__batchsize])
                    ystored[i] = splitted[1]
                    yout[i] = splitted[0]
                for i in range(0,dimw):
                    splitted=numpy.split(wstored[i],[self.__batchsize])
                    wstored[i] = splitted[1]
                    wout[i] = splitted[0]
                    
                
            processedbatches+=1
            
            #print('\n')
            #print(wout[0])
            if False and self.useweights:
                idxs=numpy.where(yout[0][:,2] == 1)
                wout[0][idxs] *= 0.5
            #print(wout[0])
            
            if self.useweights:
                yield (xout,yout,wout)
            else:
                yield (xout,yout)
            
            

    
    
    
    