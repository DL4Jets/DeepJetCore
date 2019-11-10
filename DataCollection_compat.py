'''
Created on 21 Feb 2017

@author: jkiesele
'''
#from tensorflow.contrib.labeled_tensor import batch
#from builtins import list
from __future__ import print_function

import os
import copy
import pickle
import time
import tempfile
import shutil
from stopwatch import stopwatch
import numpy as np
from Weighter import Weighter
from TrainData import TrainData, fileTimeOut
#for convenience
import logging
from pdb import set_trace

usenewformat=True

logger = logging.getLogger(__name__)

# super not-generic without safety belts
#needs some revision
class BatchRandomInputGenerator(object):
    def __init__(self, ranges, batchsize):
        self.ranges=ranges
        self.batchsize=batchsize
        
    def generateBatch(self):
        randoms=[]
        for i in range(len(self.ranges)):
            randoms.append(np.full((1,self.batchsize),np.random.uniform(self.ranges[i][0], self.ranges[i][1], size=1)[0]))
        
        nparr=np.dstack((randoms))
        return nparr.reshape(nparr.shape[1],nparr.shape[2])

class DataCollection(object):
    '''
    classdocs
    '''


    def __init__(self, infile = None, nprocs = -1):
        '''
        Constructor
        '''
        self.clear()
        self.nprocs = nprocs       
        self.meansnormslimit=500000 
        if infile:
            self.readFromFile(infile)
            #check for consistency
            if not len(self.samples):
                raise Exception("no valid datacollection found in "+infile)

        # Running data conversion etc. on a batch farm
        self.batch_mode = False
        self.no_copy_on_convert = False
        
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
        self.maxFilesOpen=2
        self.means=None
        self.classweights={}

    def __len__(self):
        return self.nsamples

    def __iadd__(self, other):
        'A += B'
        if not isinstance(other, DataCollection):
            raise ValueError("I don't know how to add DataCollection and %s" % type(other))
        def _extend_(a, b, name):
            getattr(a, name).extend(getattr(b, name))
        _extend_(self, other, 'samples')
        if len(set(self.samples)) != len(self.samples):
            raise ValueError('The two DataCollections being summed contain the same files!')
        _extend_(self, other, 'sampleentries')
        _extend_(self, other, 'originRoots')
        self.nsamples += other.nsamples
        if self.dataDir != other.dataDir:
            raise ValueError('The two DataCollections have different data directories, still to be implemented!')
        self.useweights = self.useweights and self.useweights
        self.filesPreRead = min(self.filesPreRead, other.filesPreRead)
        self.isTrain = self.isTrain and other.isTrain #arbitrary choice, could also raise exception
        if type(self.dataclass) != type(other.dataclass):
            raise ValueError(
                'The two DataCollections were made with a'
                ' different data class type! (%s, and %s)' % (type(self.dataclass), type(other.dataclass))
                )
        if self.weighter != other.weighter:
            raise ValueError(
                'The two DataCollections have different weights'
                )
        if self.weightsfraction != other.weightsfraction:
            raise ValueError('The two DataCollections have different weight fractions')
        self.maxConvertThreads = min(self.maxConvertThreads, other.maxConvertThreads)
        self.maxFilesOpen = min(self.maxFilesOpen, other.maxFilesOpen)
        if not all(self.means == other.means):
            raise ValueError(
                'The two DataCollections head different means'
                )
        self.classweights.update(other.classweights)
        return self

    def __add__(self, other):
        'A+B'
        if not isinstance(other, DataCollection):
            raise ValueError("I don't know how to add DataCollection and %s" % type(other))
        ret = copy.deepcopy(self)
        ret += other
        return ret

    def __radd__(self, other):
        'B+A to work with sum'
        if other == 0:
            return copy.deepcopy(self)
        elif isinstance(other, DataCollection):
            return self + other #we use the __add__ method
        else:
            raise ValueError("I don't know how to add DataCollection and %s" % type(other))
        
    def removeLast(self):
        self.samples.pop()
        self.nsamples-=self.sampleentries[-1]
        self.sampleentries.pop()
        self.originRoots.pop()
        
        
    def getClassWeights(self):
        if not len(self.classweights):
            self.__computeClassWeights(self.dataclass.getUsedTruth())
        return self.classweights
        
    def __computeClassWeights(self,truthclassesarray):
        if not len(self.samples):
            raise Exception("DataCollection:computeClassWeights: no sample files associated")
        td=copy.deepcopy(self.dataclass)
        td.readIn(self.getSamplePath(self.samples[0]))
        arr=td.y[0]
        average=0
        allist=[]
        for i in range(arr.shape[1]):
            entries=float((arr[:,i]>0).sum())
            average=average+entries
            allist.append(entries)
        
        outdict={}
        average=average/float((arr.shape[1]))
        for i in range(len(allist)):
            l=average/allist[i] 
            outdict[i]=l
        self.classweights=outdict
        
    
    def prependToSampleFiles(self, path_to_prepend):
        newsamples=[]
        for s in self.samples:
            newsamples.append(path_to_prepend+s)
        self.samples=newsamples
            
    def defineCustomPredictionLabels(self, labels):
        self.dataclass.defineCustomPredictionLabels(labels)
        
    
    def getCustomPredictionLabels(self):
        if hasattr(self.dataclass, 'customlabels'):
            return self.dataclass.customlabels
        return None
        
    def getInputShapes(self):
        '''
        gets the input shapes from the data class description
        '''
        if len(self.samples)<1:
            return []
        self.dataclass.filelock=None
        td=copy.deepcopy(self.dataclass)
        td.readIn(self.getSamplePath(self.samples[0]),shapesOnly=True)
        shapes=td.getInputShapes()
        td.clear()
        return shapes
    
    def getTruthShape(self):
        return self.dataclass.getTruthShapes()
        
    def getNRegressionTargets(self):
        return (self.dataclass.getNRegressionTargets())
    
    def getNClassificationTargets(self):
        return (self.dataclass.getNClassificationTargets())
        
    def getUsedTruth(self):
        return self.dataclass.getUsedTruth()
    
    def setBatchSize(self,bsize):
        if bsize > self.nsamples:
            raise Exception('Batch size must not be bigger than total sample size')
        self.__batchsize=bsize

    def getBatchSize(self):
        return self.__batchsize
        
    @property
    def batch_size(self):
        return self.__batchsize

    def getSamplesPerEpoch(self):
        #modify by batch split
        count=self.getNBatchesPerEpoch()
        if count != 1:
            return count*self.__batchsize #final
        else:
            return self.nsamples
        
    def getAvEntriesPerFile(self):
        return float(self.nsamples)/float(len(self.samples))
        
    
    def getNBatchesPerEpoch(self):
        if self.__batchsize <= 1:
            return 1
        count=0
        while (count+1)*self.__batchsize <= self.nsamples:
            count+=1
        return count
    
    def validate(self, remove=True, skip_first=0):
        '''
        checks if all samples in the collection can be read properly.
        removes the invalid samples from the sample list.
        Also removes the original link to the root file, so recover cannot be run
        (this might be changed in future implementations)
        '''
        for i in range(len(self.samples)):
            if i < skip_first: continue
            if i >= len(self.samples): break
            td=copy.deepcopy(self.dataclass)
            fullpath=self.getSamplePath(self.samples[i])
            print('reading '+fullpath, str(self.sampleentries[i]), str(i), '/', str(len(self.samples)))
            try:
                td.readIn(fullpath)
                for x in td.x:
                    if td.nsamples != x.shape[0]:
                        print("not right length")
                        raise Exception("not right length")
                for y in td.y:
                    if td.nsamples != y.shape[0]:
                        print("not right length")
                        raise Exception("not right length")
                
                del td
                continue
            except Exception as e:
                print('problem with file, removing ', fullpath)
                del self.samples[i]
                del self.originRoots[i]
                self.nsamples -= self.sampleentries[i]
                del self.sampleentries[i]
                
    def removeEntry(self,relative_path_to_entry):
        for i in range(len(self.samples)):
            if relative_path_to_entry==self.samples[i]:
                print('removing '+self.samples[i]+" - "+str(self.sampleentries[i]))
                del self.samples[i]
                del self.originRoots[i]
                self.nsamples -= self.sampleentries[i]
                del self.sampleentries[i]
                break
                 
        
    def writeToFile(self,filename):
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as fd:
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

        shutil.move(fd.name, filename)
        
    ## for conversion essential!!!
    def readRawFromFile(self,filename):
        #no assumption on data class
        fd=open(filename,'rb')
        self.samples=pickle.load(fd)
        self.sampleentries=pickle.load(fd)
        self.originRoots=pickle.load(fd)
        fd.close()
        
    def readFromFile(self,filename):
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

        self.dataDir=os.path.dirname(os.path.abspath(filename))
        self.dataDir+='/'
        #don't check if files exist
        return 
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
        
        
    def readRootListFromFile(self, file, relpath=''):
        self.samples=[]
        self.sampleentries=[]
        self.originRoots=[]
        self.nsamples=0
        self.dataDir=""
        
        fdir=os.path.dirname(file)
        fdir=os.path.abspath(fdir)
        fdir=os.path.realpath(fdir)
        lines = [line.rstrip('\n') for line in open(file)]
        for line in lines:
            if len(line) < 1: continue
            if relpath:
                self.originRoots.append(os.path.join(relpath, line))
            else:
                self.originRoots.append(line)

        if len(self.originRoots)<1:
            raise Exception('root samples list empty')
        
        
    def split(self,ratio):
        '''
        ratio is self/(out+self)
        returns out
        modifies itself
        '''
        
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
        out.useweights=self.useweights
     
        
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
    
    
    def createTestDataForDataCollection(self, collectionfile, inputfile, outputDir,
            outname='dataCollection.dc',
            traind=None,
            relpath=''):

        self.readFromFile(collectionfile)
        self.dataclass.remove=False
        self.dataclass.weight=True #False
        if traind: 
            print('[createTestDataForDataCollection] dataclass is overriden by user request')
            self.dataclass=traind
        self.readRootListFromFile(inputfile, relpath=relpath)
        self.createDataFromRoot(
            self.dataclass, outputDir, False,
            dir_check = not self.batch_mode
        )
        self.writeToFile(outputDir+'/'+outname)
        
    
    def recoverCreateDataFromRootFromSnapshot(self, snapshotfile):
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
        self.writeToFile(outputDir+'/dataCollection.dc')
        
    
    def createDataFromRoot(
                    self, dataclass, outputDir, 
                    redo_meansandweights=True, means_only=False, dir_check=True
                    ):
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
        
        outputDir+='/'
        if os.path.isdir(outputDir) and dir_check:
            raise Exception('output dir must not exist')
        elif not os.path.isdir(outputDir):
            os.mkdir(outputDir)
        self.dataDir=outputDir
        self.nsamples=0
        self.samples=[]
        self.sampleentries=[]
        self.dataclass=copy.deepcopy(dataclass)
        td=self.dataclass
        ##produce weighter from a larger dataset as one file

       
        if redo_meansandweights and (td.remove or td.weight):
            logging.info('producing weights and remove indices')
            self.weighter = td.produceBinWeighter(
                self.originRoots
                )            
            self.weighter.printHistos(outputDir)
            
        
        if redo_meansandweights:
            logging.info('producing means and norms')
            self.means = td.produceMeansFromRootFile(
                self.originRoots, limit=self.meansnormslimit
                )
        
        if means_only: return

        if self.batch_mode:
            for sample in self.originRoots:
                self.__writeData(sample, outputDir)
        else:
            self.__writeData_async_andCollect(0, outputDir)
    
    def __writeData(self, sample, outputDir):
        sw=stopwatch()
        td=copy.deepcopy(self.dataclass)
        
        fileTimeOut(sample,120) #once available copy to ram

        if self.batch_mode or self.no_copy_on_convert:
            tmpinput = sample

            def removefile():
                pass
        else:
            tmpinput = '/dev/shm/'+str(os.getpid())+os.path.basename(sample)
            
            def removefile():
                os.system('rm -f '+tmpinput)
            
            import atexit
            atexit.register(removefile)
            
            os_ret = os.system('cp '+sample+' '+tmpinput)
            if os_ret:
                raise Exception("copy to ramdisk not successful for "+sample)

        try:
            td.readFromRootFile(tmpinput, self.means, self.weighter)
            sbasename = os.path.basename(sample)
            newname = sbasename[:sbasename.rfind('.')]
            
            if usenewformat:
                newname+='.meta'
            else:
                newname+='.z'
            newpath=os.path.abspath(outputDir+newname)
            td.writeOut(newpath)
            print('converted and written '+newname+' in ',sw.getAndReset(),' sec')
            self.samples.append(newname)
            self.nsamples+=td.nsamples
            self.sampleentries.append(td.nsamples)
            td.clear()

            if not self.batch_mode:
                self.writeToFile(outputDir+'/snapshot.dc')
                
        finally:
            removefile()
        
    def __writeData_async_andCollect(self, startindex, outputDir):
        
        #set tree name to use
        logger.info('setTreeName')
        import DeepJetCore.preprocessing
        DeepJetCore.preprocessing.setTreeName(self.dataclass.treename)
        
        from multiprocessing import Process, Queue, cpu_count, Lock
        wo_queue = Queue()
        writelock=Lock()
        thispid=str(os.getpid())
        if not self.batch_mode and not os.path.isfile(outputDir+'/snapshot.dc'):
            self.writeToFile(outputDir+'/snapshot.dc')
        
        tempstoragepath='/dev/shm/'+thispid
        
        logger.info('creating dir '+tempstoragepath)
        os.system('mkdir -p '+tempstoragepath)
        
        def writeData_async(index,woq,wrlck):

            logger.info('async started')
            
            sw=stopwatch()
            td=copy.deepcopy(self.dataclass)
            sample=self.originRoots[index]

            if self.batch_mode or self.no_copy_on_convert:
                tmpinput = sample

                def removefile():
                    pass
            else:
                tmpinput = tempstoragepath+'/'+str(os.getpid())+os.path.basename(sample)
                
                def removefile():
                    os.system('rm -f '+tmpinput)
                
                import atexit
                atexit.register(removefile)

                logger.info('start cp')
                os_ret=os.system('cp '+sample+' '+tmpinput)
                if os_ret:
                    raise Exception("copy to ramdisk not successful for "+sample)
                
            success=False
            out_samplename=''
            out_sampleentries=0
            sbasename = os.path.basename(sample)
            newname = sbasename[:sbasename.rfind('.')]
            if usenewformat:
                newname+='.meta'
            else:
                newname+='.z'
            newpath=os.path.abspath(outputDir+newname)
            
            try:
                logger.info('readFromRootFile')
                td.readFromRootFile(tmpinput, self.means, self.weighter)
                logger.info('writeOut')
                #wrlck.acquire()
                td.writeOut(newpath)
                #wrlck.release()
                print('converted and written '+newname+' in ',sw.getAndReset(),' sec -', index)
                
                out_samplename=newname
                out_sampleentries=td.nsamples
                success=True
                td.clear()
                removefile()
                woq.put((index,[success,out_samplename,out_sampleentries]))
                
            except:
                print('problem in '+newname)
                removefile()
                woq.put((index,[False,out_samplename,out_sampleentries]))
                raise 
        
        def __collectWriteInfo(successful,samplename,sampleentries,outputDir):
            if not successful:
                raise Exception("write not successful, stopping")

            self.samples.append(samplename)
            self.nsamples+=sampleentries
            self.sampleentries.append(sampleentries)
            if not self.batch_mode:
                self.writeToFile(outputDir+'/snapshot_tmp.dc')#avoid to overwrite directly
                os.system('mv '+outputDir+'/snapshot_tmp.dc '+outputDir+'/snapshot.dc')
            
        processes=[]
        processrunning=[]
        processfinished=[]
        for i in range(startindex,len(self.originRoots)):
            processes.append(Process(target=writeData_async, args=(i,wo_queue,writelock) ) )
            processrunning.append(False)
            processfinished.append(False)
        
        nchilds = int(cpu_count()/2)-2 if self.nprocs <= 0 else self.nprocs
        #if 'nvidiagtx1080' in os.getenv('HOSTNAME'):
        #    nchilds=cpu_count()-5
        if nchilds<1: 
            nchilds=1
        
        #nchilds=10
        
        
        
        lastindex=startindex-1
        alldone=False
        results=[]

        try:
            while not alldone:
                nrunning=0
                for runs in processrunning:
                    if runs: nrunning+=1
                
                for i in range(len(processes)):
                    if nrunning>=nchilds:
                        break
                    if processrunning[i]:continue
                    if processfinished[i]:continue
                    time.sleep(0.1)
                    logging.info('starting %s...' % self.originRoots[startindex+i])
                    processes[i].start()
                    processrunning[i]=True
                    nrunning+=1
                    
                
                
                if not wo_queue.empty():
                    res=wo_queue.get()
                    results.append(res)
                    originrootindex=res[0]
                    logging.info('finished %s...' % self.originRoots[originrootindex])
                    processfinished[originrootindex-startindex]=True
                    processes      [originrootindex-startindex].join(5)
                    processrunning [originrootindex-startindex]=False  
                    #immediately send the next
                    continue
                  
                results = sorted(results, key=lambda x:x[0])    
                for r in results:
                    thisidx=r[0]
                    if thisidx==lastindex+1:
                        logging.info('>>>> collected result %d of %d' % (thisidx+1,len(self.originRoots)))
                        __collectWriteInfo(r[1][0],r[1][1],r[1][2],outputDir)
                        lastindex=thisidx        
                
                if nrunning==0:
                    alldone=True
                    continue
                time.sleep(0.1)
                  
        except:
            os.system('rm -rf '+tempstoragepath)
            raise 
        os.system('rm -rf '+tempstoragepath)
        
    def convertListOfRootFiles(self, inputfile, dataclass, outputDir, 
            takemeansfrom='', means_only=False,
            output_name='dataCollection.dc',
            relpath=''):
        
        newmeans=True
        if takemeansfrom:
            self.readFromFile(takemeansfrom)
            newmeans=False
        self.readRootListFromFile(inputfile, relpath=relpath)
        self.createDataFromRoot(
                    dataclass, outputDir, 
                    newmeans, means_only = means_only, 
                    dir_check= not self.batch_mode
                    )
        self.writeToFile(outputDir+'/'+output_name)
        
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
                        out[i] = np.append(out[i],thislist[i])
                    else:
                        out[i] = np.vstack((out[i],thislist[i]))
                
        return out
    
        
    def replaceTruthForGAN(self, generated_array, original_truth):
        return self.dataclass.replaceTruthForGAN(generated_array, original_truth)
        
    def generator(self):
        from sklearn.utils import shuffle
        import uuid
        import threading
        print('start generator')
        #helper class
        class tdreader(object):
            def __init__(self,filelist,maxopen,tdclass):
                
                self.filelist=filelist
                self.nfiles=len(filelist)
                
                self.tdlist=[]
                self.tdopen=[]
                self.tdclass=copy.deepcopy(tdclass)
                self.tdclass.clear()#only use the format, no data
                #self.copylock=thread.allocate_lock()
                for i in range(self.nfiles):
                    self.tdlist.append(copy.deepcopy(tdclass))
                    self.tdopen.append(False)
                    
                self.closeAll() #reset state
                self.shuffleseed=0
                
            def start(self):
                self.__readNext()
                    
            def __readNext(self):
                #make sure this fast function has exited before getLast tries to read the file
                readfilename=self.filelist[self.filecounter]
                if len(filelist)>1:
                    self.tdlist[self.nextcounter].clear()
                self.tdlist[self.nextcounter]=copy.deepcopy(self.tdclass)
                self.tdlist[self.nextcounter].readthread=None
                
                def startRead(counter,filename,shuffleseed):   
                    excounter=0
                    while excounter<10:
                        try:
                            self.tdlist[counter].readIn_async(filename,ramdiskpath='/dev/shm/',
                                                              randomseed=shuffleseed)
                            break
                        except Exception as d:
                            print(self.filelist[counter]+' read error, retry...')
                            self.tdlist[counter].readIn_abort()
                            excounter=excounter+1
                            if excounter<10:
                                time.sleep(5)
                                continue
                            traceback.print_exc(file=sys.stdout)
                            raise d
                    
                # don't remove these commented lines just yet 
                # the whole generator call is moved to thread since keras 2.0.6 anyway  
                #t=threading.Thread(target=startRead, args=(self.nextcounter,readfilename,self.shuffleseed))    
                #t.start()
                startRead(self.nextcounter,readfilename,self.shuffleseed)
                self.shuffleseed+=1
                if self.shuffleseed>1e5:
                    self.shuffleseed=0
                #startRead(self.nextcounter,readfilename,self.shuffleseed)
                self.tdopen[self.nextcounter]=True
                self.filecounter=self.__increment(self.filecounter,self.nfiles,to_shuffle=True)
                self.nextcounter=self.__increment(self.nextcounter,self.nfiles)
                
                
                
            def __getLast(self):
                #print('joining...') #DEBUG PERF
                self.tdlist[self.lastcounter].readIn_join(wasasync=True,waitforStart=True)
                #print('joined') #DEBUG PERF
                td=self.tdlist[self.lastcounter]
                #print('got ',self.lastcounter)
                
                self.tdopen[self.lastcounter]=False
                self.lastcounter=self.__increment(self.lastcounter,self.nfiles)
                return td
                
            def __increment(self,counter,maxval,to_shuffle=False):
                counter+=1
                if counter>=maxval:
                    counter=0
                    if to_shuffle:
                        self.filelist = shuffle(self.filelist)
                return counter 
            
            def __del__(self):
                self.closeAll()
                
            def closeAll(self):
                for i in range(len(self.tdopen)):
                    try:
                        if self.tdopen[i]:
                            self.tdlist[i].readIn_abort()
                            self.tdlist[i].clear()
                            self.tdopen[i]=False
                    except: pass
                    self.tdlist[i].removeRamDiskFile()
                
                self.nextcounter=0
                self.lastcounter=0
                self.filecounter=0
                
            def get(self):
                
                td=self.__getLast()
                self.__readNext()
                return td
                
        
        td=(self.dataclass)
        totalbatches=self.getNBatchesPerEpoch()
        processedbatches=0
        
        ####generate randoms by batch
        batchgen=None
        if hasattr(td,'generatePerBatch') and td.generatePerBatch:
            ranges=td.generatePerBatch
            batchgen=BatchRandomInputGenerator(ranges, self.__batchsize)
        
        xstored=[np.array([])]
        dimx=0
        ystored=[]
        dimy=0
        wstored=[]
        dimw=0
        nextfiletoread=0
        
        target_xlistlength=len(td.getInputShapes())
        
        xout=[]
        yout=[]
        wout=[]
        samplefilecounter=0
        
        #prepare file list
        filelist=[]
        for s in self.samples:
            filelist.append(self.getSamplePath(s))
        
        TDReader=tdreader(filelist, self.maxFilesOpen, self.dataclass)
        
        #print('generator: total batches '+str(totalbatches))
        print('start file buffering...')
        TDReader.start()
        #### 
        #
        # make block class for file read with get function that starts the next read automatically
        # and closes all files in destructor?
        #
        #  check if really the right ones are read....
        #
        psamples=0 #for random shuffling
        nepoch=0
        shufflecounter=0
        shufflecounter2=0
        while 1:
            if processedbatches == totalbatches:
                processedbatches=0
                nepoch+=1
                shufflecounter2+=1
            
            lastbatchrest=0
            if processedbatches == 0: #reset buffer and start new
                #print('DataCollection: new turnaround')
                xstored=[np.array([])]
                dimx=0
                ystored=[]
                dimy=0
                wstored=[]
                dimw=0
                lastbatchrest=0
                
            
            else:
                lastbatchrest=xstored[0].shape[0]
            
            batchcomplete=False   
            
            if lastbatchrest >= self.__batchsize:
                batchcomplete = True
                
            # if(xstored[1].ndim==1):
                
            while not batchcomplete:
                import sys, traceback
                try:
                    td=TDReader.get()
                except:
                    traceback.print_exc(file=sys.stdout)
                    
                if td.x[0].shape[0] == 0:
                    print('Found empty (corrupted?) file, skipping')
                    continue
                
                if xstored[0].shape[0] ==0:
                    #print('dc:read direct') #DEBUG
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
                    
                    #randomly^2 shuffle - not needed every time
                    if shufflecounter>1 and shufflecounter2>1:
                        shufflecounter=0
                        shufflecounter2=0
                        for i in range(0,dimx):
                            td.x[i]=shuffle(td.x[i], random_state=psamples)
                        for i in range(0,dimy):
                            td.y[i]=shuffle(td.y[i], random_state=psamples)
                        for i in range(0,dimw):
                            td.w[i]=shuffle(td.w[i], random_state=psamples)
                    
                    shufflecounter+=1
                    
                    for i in range(0,dimx):
                        if(xstored[i].ndim==1):
                            xstored[i] = np.append(xstored[i],td.x[i])
                        else:
                            xstored[i] = np.vstack((xstored[i],td.x[i]))
                    
                    for i in range(0,dimy):
                        if(ystored[i].ndim==1):
                            ystored[i] = np.append(ystored[i],td.y[i])
                        else:
                            ystored[i] = np.vstack((ystored[i],td.y[i]))
                    
                    for i in range(0,dimw):
                        if(wstored[i].ndim==1):
                            wstored[i] = np.append(wstored[i],td.w[i])
                        else:
                            wstored[i] = np.vstack((wstored[i],td.w[i]))
                    
                if xstored[0].shape[0] >= self.__batchsize:
                    batchcomplete = True
                    
                #limit of the random generator number 
                psamples+=  td.x[0].shape[0]   
                if psamples > 4e8:
                    psamples/=1e6
                    psamples=int(psamples)
                
                td.clear()
                
            if batchcomplete:
                
                #print('batch complete, split')#DEBUG
                
                for i in range(0,dimx):
                    splitted=np.split(xstored[i],[self.__batchsize])
                    xstored[i] = splitted[1]
                    xout[i] = splitted[0]
                for i in range(0,dimy):
                    splitted=np.split(ystored[i],[self.__batchsize])
                    ystored[i] = splitted[1]
                    yout[i] = splitted[0]
                for i in range(0,dimw):
                    splitted=np.split(wstored[i],[self.__batchsize])
                    wstored[i] = splitted[1]
                    wout[i] = splitted[0]
            
            for i in range(0,dimx):
                if(xout[i].ndim==1):
                    xout[i]=(xout[i].reshape(xout[i].shape[0],1)) 
                if not xout[i].shape[1] >0:
                    raise Exception('serious problem with the output shapes!!')
                            
            for i in range(0,dimy):
                if(yout[i].ndim==1):
                    yout[i]=(yout[i].reshape(yout[i].shape[0],1))
                if not yout[i].shape[1] >0:
                    raise Exception('serious problem with the output shapes!!')
                    
            for i in range(0,dimw):
                if(wout[i].ndim==1):
                    wout[i]=(wout[i].reshape(wout[i].shape[0],1))
                if not xout[i].shape[1] >0:
                    raise Exception('serious problem with the output shapes!!')
            
            processedbatches+=1
            
            
            if batchgen:
                if len(xout)<target_xlistlength:
                    xout.append(batchgen.generateBatch())
                else:
                    xout[-1]=batchgen.generateBatch()
                    
            if self.useweights:
                yield (xout,yout,wout)
            else:
                yield (xout,yout)
            
            

    
    
    
    
