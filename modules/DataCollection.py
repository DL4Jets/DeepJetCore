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
        self.sampleentries=[]
        self.originRoots=[]
        self.nsamples=0
        self.dataDir=""
        self.useweights=True
        self.__batchsize=1
        self.filesPreRead=2
        self.isTrain=True
        self.dataclass=TrainData() #for future implementations
        self.weighter=Weighter()
        self.weightsfraction=0.05
        self.maxConvertThreads=2
        self.maxFilesOpen=3
        self.means=None
        self.classweights={}
        
    def removeLast(self):
        self.samples.pop()
        self.nsamples-=self.sampleentries[-1]
        self.sampleentries.pop()
        self.originRoots.pop()
        
        
    def getClassWeights(self):
        return 0 #TBI
        
    def __computeClassWeights(self,truthclassesarray):
        return 0 #TBI
        
    def getInputShapes(self):
        '''
        gets the input shapes from the data class description
        '''
        import copy
        if len(self.samples)<1:
            return []
        self.dataclass.filelock=None
        td=copy.deepcopy(self.dataclass)
        td.readIn(self.getSamplePath(self.samples[0]))
        shapes=td.getInputShapes()
        td.clear()
        return shapes
    
    def getTruthShape(self):
        td=self.dataclass
        return td.getTruthShapes()
        
    
        
    def getUsedTruth(self):
        return self.dataclass.getUsedTruth()
    
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
        self.dataclass.clear()
        pickle.dump(self.samples, fd,protocol=0 )
        pickle.dump(self.sampleentries, fd,protocol=0 )
        pickle.dump(self.originRoots, fd,protocol=0 )
        pickle.dump(self.nsamples, fd,protocol=0 )
        pickle.dump(self.useweights, fd,protocol=0 )
        pickle.dump(self.__batchsize, fd,protocol=0 )
        pickle.dump(self.dataclass, fd,protocol=0 )
        pickle.dump(self.weighter, fd,protocol=0 )
        #pickle.dump(self.means, fd,protocol=0 )
        self.means.dump(fd)
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
        self.dataclass=pickle.load(fd)
        self.weighter=pickle.load(fd)
        self.means=pickle.load(fd)
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
        
        self.samples=[]
        self.sampleentries=[]
        self.originRoots=[]
        self.nsamples=0
        self.dataDir=""
        
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
        import copy
        
        
        out=DataCollection()
        itself=copy.deepcopy(self)
        
        nsamplefiles=len(self.samples)
        
        out.samples=[]
        out.sampleentries=[]
        out.originRoots=[]
        out.nsamples=0
        out.__batchsize=copy.deepcopy(self.__batchsize)
        out.isTrain=copy.deepcopy(self.isTrain)
        out.dataDir=self.dataDir

        out.dataclass=copy.deepcopy(self.dataclass)
        out.weighter=self.weighter #ref oks
        out.means=self.means
     
        
        itself.samples=[]
        itself.sampleentries=[]
        itself.originRoots=[]
        itself.nsamples=0
        
        
        
        if nsamplefiles < 2:
            out=copy.deepcopy(self)
            print('DataCollection.split: warning: only one file, split will just return a copy of this')
            return out
    
        for i in range(0, nsamplefiles):
            frac=(float(i))/(float(nsamplefiles))
            if frac < ratio and i < nsamplefiles-1:
                itself.samples.append(self.samples[i])
                itself.sampleentries.append(self.sampleentries[i])
                itself.originRoots.append(self.originRoots[i])
                itself.nsamples+=self.sampleentries[i]
            else:
                out.samples.append(self.samples[i])
                out.sampleentries.append(self.sampleentries[i])
                out.originRoots.append(self.originRoots[i])
                out.nsamples+=self.sampleentries[i]
           
        
        
        self.samples=itself.samples
        self.sampleentries=itself.sampleentries
        self.originRoots=itself.originRoots
        self.nsamples=itself.nsamples
        
        return out
    
    
    def createTestDataForDataCollection(self,collectionfile,inputfile,outputDir):
        import copy
        self.readFromFile(collectionfile)
        self.dataclass.remove=False
        self.dataclass.weight=False
        self.readRootListFromFile(inputfile)
        self.createDataFromRoot(self.dataclass, outputDir,False)
        self.writeToFile(outputDir+'/dataCollection.dc')
        
        
    
    def recoverCreateDataFromRootFromSnapshot(self, snapshotfile):
        import os
        snapshotfile=os.path.abspath(snapshotfile)
        self.readFromFile(snapshotfile)
        td=self.dataclass
        #For emergency recover  td.reducedtruthclasses=['isB','isC','isUDSG']
        if len(self.originRoots) < 1:
            return
        #if not self.means:
        #    self.means=td.produceMeansFromRootFile(self.originRoots[0])
        outputDir=os.path.dirname(snapshotfile)+'/'
        self.dataDir=outputDir
        finishedsamples=len(self.samples)
        
        self.__writeData_async_andCollect(finishedsamples,outputDir)
        
    
    def createDataFromRoot(self,dataclass, outputDir, redo_meansandweights=True):
        '''
        Also creates a file list of the output files
        After the operation, the object will point to the already processed
        files (not root files)
        Writes out a snapshot of itself after every successfully written output file
        to recover the data until a possible error occurred
        '''
        
        if len(self.originRoots) < 1:
            print('createDataFromRoot: no input root file')
            raise Exception('createDataFromRoot: no input root file')
        
        import os
        outputDir+='/'
        if os.path.isdir(outputDir):
            raise Exception('output dir must not exist')
        os.mkdir(outputDir)
        self.dataDir=outputDir
        self.nsamples=0
        self.samples=[]
        self.sampleentries=[]
        import copy
        self.dataclass=copy.deepcopy(dataclass)
        td=self.dataclass
        ##produce weighter from a larger dataset as one file
        
        
        if redo_meansandweights and (td.remove or td.weight):
            print('producing weights')
            weighter=Weighter()
            weighter=td.produceBinWeighter(self.originRoots[0])
            self.weighter=weighter
        
        if redo_meansandweights:
            print('producing means')
            self.means=td.produceMeansFromRootFile(self.originRoots[0])
        
        
        self.__writeData_async_andCollect(0,outputDir)
        
        
    
    def __writeData(self,sample,means, weighter,outputDir,dataclass):
        import os
        import copy
        from stopwatch import stopwatch
        sw=stopwatch()
        td=copy.deepcopy(dataclass)
        
        td.fileTimeOut(sample,120) #once available copy to ram
        ramdisksample= '/dev/shm/'+str(os.getpid())+os.path.basename(sample)
        
        def removefile():
            os.system('rm -f '+ramdisksample)
        
        import atexit
        atexit.register(removefile)
        
        os.system('cp '+sample+' '+ramdisksample)
        try:
            td.readFromRootFile(ramdisksample,means, weighter) 
            newname=os.path.basename(sample).rsplit('.', 1)[0]
            newpath=os.path.abspath(outputDir+newname+'.z')
            td.writeOut(newpath)
            print('converted and written '+newname+'.z in ',sw.getAndReset(),' sec')
            self.samples.append(newname+'.z')
            self.nsamples+=td.nsamples
            self.sampleentries.append(td.nsamples)
            td.clear()
            self.writeToFile(outputDir+'/snapshot.dc')
        except Exception as e:
            removefile()
            raise e
        removefile()
        
        
    def __writeData_async_andCollect(self, startindex, outputDir):
        
        from multiprocessing import Process, Queue, cpu_count
        wo_queue = Queue()
        
        def writeData_async(index,woq):
            import os
            import copy
            from stopwatch import stopwatch
            sw=stopwatch()
            td=copy.deepcopy(self.dataclass)
            sample=self.originRoots[index]
            td.fileTimeOut(sample,120) #once available copy to ram
            ramdisksample= '/dev/shm/'+str(os.getpid())+os.path.basename(sample)
            
            def removefile():
                os.system('rm -f '+ramdisksample)
            
            import atexit
            atexit.register(removefile)
            success=False
            out_samplename=''
            out_sampleentries=0
            os.system('cp '+sample+' '+ramdisksample)
            try:
                td.readFromRootFile(ramdisksample,self.means, self.weighter) 
                newname=os.path.basename(sample).rsplit('.', 1)[0]
                newpath=os.path.abspath(outputDir+newname+'.z')
                td.writeOut(newpath)
                print('converted and written '+newname+'.z in ',sw.getAndReset(),' sec')
                
                out_samplename=newname+'.z'
                out_sampleentries=td.nsamples
                success=True
                td.clear()
                
                #this goes after join
                
            except Exception as e:
                removefile()
                raise e
            removefile()
            woq.put((index,[success,out_samplename,out_sampleentries]))
        
        
        def __collectWriteInfo(successful,samplename,sampleentries,outputDir):
            if not successful:
                raise Exception("write not successful, stopping")
            
            self.samples.append(samplename)
            self.nsamples+=sampleentries
            self.sampleentries.append(sampleentries)
            self.writeToFile(outputDir+'/snapshot.dc')
            
        processes=[]
        for i in range(startindex,len(self.originRoots)):
            processes.append(Process(target=writeData_async, args=(i,wo_queue) ) )
        
        nchilds=cpu_count()-4 #don't use all of them
        #import os
        #if 'nvidiagtx1080' in os.getenv('HOSTNAME'):
        #    nchilds=cpu_count()-5
        if nchilds<1: 
            nchilds=1
        
        index=0
        alldone=False
        while not alldone:
            if index+nchilds >= len(self.originRoots):
                nchilds=len(self.originRoots)-index
                alldone=True
                
            for i in range(nchilds):
                processes[i+index].start()
            for i in range(nchilds):
                processes[i+index].join()
            results = [wo_queue.get() for i in range(nchilds)]
            results.sort()
            results = [r[1] for r in results]
            for i in range(nchilds):
                print(results[i])
                __collectWriteInfo(results[i][0],results[i][1],results[i][2],outputDir)
            
            index+=nchilds
        
        
    def convertListOfRootFiles(self, inputfile, dataclass, outputDir):
        self.readRootListFromFile(inputfile)
        self.createDataFromRoot(dataclass, outputDir)
        self.writeToFile(outputDir+'/dataCollection.dc')
        
    def getAllLabels(self):
        return self.__stackData(self.dataclass,'y')
    
    def getAllFeatures(self):
        return self.__stackData(self.dataclass,'x')
        
    def getAllWeights(self):
        return self.__stackData(self.dataclass,'w')
    
    
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
    
        
    
        
    def generator(self):
        import numpy
        
        td=(self.dataclass)
        totalbatches=self.getNBatchesPerEpoch()
        processedbatches=0
        
        #print(totalbatches,self.__batchsize,self.nsamples)
        
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
        samplefilecounter=0
        
        import copy
        ttdl=[]
        nsamplesinqueue=0
        for s in self.samples:
            ttdl.append(copy.deepcopy(td))
        
        
        #locks=[]
        #from threading import Lock
        #for i in range(len(self.samples)):
        #    locks.append(Lock())
        ##read the next sample in advance
        #from threading import Thread
        #def readTDThread(idx,TD):
        #    locks[idx].acquire()
        #    print('read',idx)
        #    TD.readIn(self.getSamplePath(self.samples[idx]))
        #    locks[idx].release()
            #the reading itself uses multiple cores
            #the threads spawned here will sleep on this core until
            #the reading has finished
            #prevent access to same file at the same time
            

        while 1:
            
            if processedbatches == totalbatches:
                processedbatches=0
            
            lastbatchrest=0
            if processedbatches == 0: #reset buffer and start new
                nextfiletoread=0
                samplefilecounter=0
                xstored=[numpy.array([])]
                dimx=0
                ystored=[]
                dimy=0
                wstored=[]
                dimw=0
                lastbatchrest=0
                nsamplesinqueue=0
                #readthreads=[]
                #for idx in range(len(self.samples)):
                #    readthreads.append(Thread(target=readTDThread, args=(idx,ttdl[idx])))
                
            else:
                lastbatchrest=xstored[0].shape[0]
            
            batchcomplete=False
            doSplit=True
            
            
            if lastbatchrest >= self.__batchsize:
                batchcomplete = True
                
            # if(xstored[1].ndim==1):
                
            while not batchcomplete:
               
                while nsamplesinqueue < self.maxFilesOpen and nextfiletoread < len(self.samples):
                    #print('start',nextfiletoread)
                    #readthreads[nextfiletoread].start()
                    ttdl[nextfiletoread].readIn_async(self.getSamplePath(
                        self.samples[nextfiletoread]))
                    nextfiletoread+=1
                    nsamplesinqueue+=1
                
                
                #print('\njoin:',readsample,nextfiletoread,len(self.samples),self.isTrain,'\n')
                
                ttdl[samplefilecounter].readIn_join()
                td=ttdl[samplefilecounter]
                #print('joined',samplefilecounter)
                
                
                nsamplesinqueue-=1
                samplefilecounter+=1
                
                if samplefilecounter >= len(self.samples):
                    samplefilecounter=0
                #readTDThread(td,readsample)
                #td.readIn(readsample)
                #get the format right in the first read
                if samplefilecounter == 1 or xstored[0].shape[0] ==0: #either first file or nothing left, so no need to append
                    #print('\nfull in\n')
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
                    for i in range(0,dimy):
                        yout.append([])
                    for i in range(0,dimw):
                        wout.append([])
                        
                else:
                    if self.__batchsize < td.nsamples:
                        for i in range(0,dimx):
                            splitted=numpy.split(td.x[i],[self.__batchsize-lastbatchrest])
                            if(xstored[i].ndim==1):
                                xout[i] = numpy.append(xstored[i],splitted[0])
                            else:
                                xout[i] = numpy.vstack((xstored[i],splitted[0]))
                            xstored[i]= splitted[1]
                        
                        for i in range(0,dimy):
                            splitted=numpy.split(td.y[i],[self.__batchsize-lastbatchrest])
                            if(ystored[i].ndim==1):
                                yout[i] = numpy.append(ystored[i],splitted[0])
                            else:
                                yout[i] = numpy.vstack((ystored[i],splitted[0]))
                            ystored[i]= splitted[1]
                        
                        for i in range(0,dimw):
                            splitted=numpy.split(td.w[i],[self.__batchsize-lastbatchrest])
                            if(wstored[i].ndim==1):
                                wout[i] = numpy.append(wstored[i],splitted[0])
                            else:
                                wout[i] = numpy.vstack((wstored[i],splitted[0]))
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
            
            
            if self.useweights:
                yield (xout,yout,wout)
            else:
                yield (xout,yout)
            
            

    
    
    
    
