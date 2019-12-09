'''
Created on 21 Feb 2017

@author: jkiesele
'''


from DeepJetCore.TrainData import TrainData
from DeepJetCore.compiled.c_trainDataGenerator import trainDataGenerator
import tensorflow as tf
import tempfile
import pickle
import shutil
import os
import keras 
import copy
import time
import logging
from stopwatch import stopwatch
logger = logging.getLogger(__name__)


class DataCollection(object):
    '''
    classdocs
    '''
    def __init__(self, infile = None, nprocs = -1):
        '''
        Constructor
        '''
        self.clear()
        
        self.batch_uses_sum_of_squares=False
        self.gen = None
        self.__batchsize=1
        self.optionsdict={}
        
        if infile:
            self.readFromFile(infile)
            if not len(self.samples):
                raise Exception("no valid datacollection found in "+infile)
            
        self.weighterobjects={}
        self.batch_mode = False
        self.nprocs=-1
    
    def setDataClass(self, dataclass):
        self.dataclass = dataclass
        self.dataclass_instance = self.dataclass()
           
    def clear(self):
        self.samples=[]
        self.sourceList=[]
        self.dataDir=""
        self.dataclass = TrainData
        self.dataclass_instance = self.dataclass()
        self.__nsamples = 0

    def __iadd__(self, other):
        'A += B'
        if not isinstance(other, DataCollection):
            raise ValueError("I don't know how to add DataCollection and %s" % type(other))
        def _extend_(a, b, name):
            getattr(a, name).extend(getattr(b, name))
        _extend_(self, other, 'samples')
        if len(set(self.samples)) != len(self.samples):
            raise ValueError('The two DataCollections being summed contain the same files!')
        _extend_(self, other, 'sourceList')
        if self.dataDir != other.dataDir:
            raise ValueError('The two DataCollections have different data directories, still to be implemented!')
        #if type(self.dataclass) != type(other.dataclass):
        #    raise ValueError(
        #        'The two DataCollections were made with a'
        #        ' different data class type! (%s, and %s)' % (type(self.dataclass), type(other.dataclass))
        #        )
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
        
    def _readShapesIfNeeded(self):
        if len(self.samples)<1:
            return
        if self.dataclass_instance.nElements() < 1:
            self.dataclass_instance.readShapesFromFile(self.getSamplePath(self.samples[0]))
        
    def _readNTotal(self):
        if not len(self.samples):
            return 0
        gen = trainDataGenerator()
        gen.setFileList([self.dataDir+"/"+s for s in self.samples])
        return gen.getNTotal()
        
        
    def removeLast(self):
        self.samples.pop()
        self.sourceList.pop()
        
        
    def getKerasFeatureShapes(self):
        if len(self.samples)<1:
            raise Exception("DataCollection.getKerasFeatureShapes: no files")
            return []
        self._readShapesIfNeeded()
        return self.dataclass_instance.getKerasFeatureShapes()
    
    def getInputShapes(self):
        print('DataCollection:getInputShapes deprecated, use getKerasFeatureShapes ')
        return self.getKerasFeatureShapes()
    
       
    def setBatchSize(self,bsize):
        self.__batchsize=bsize

    def getBatchSize(self):
        return self.__batchsize
    

    
    def validate(self, remove=True, skip_first=0):
        '''
        checks if all samples in the collection can be read properly.
        removes the invalid samples from the sample list.
        Also removes the original link to the root file, so recover cannot be run
        (this might be changed in future implementations)
        '''
        validsourcelist = len(self.samples) == len(self.sourceList)
        newsamples=[]
        newsources=[]
        for i in range(len(self.samples)):
            if i < skip_first: continue
            td = self.dataclass ()
            fullpath=self.getSamplePath(self.samples[i])
            print('reading '+fullpath, str(i), '/', str(len(self.samples)))
            try:
                td.readFromFile(fullpath)
                if td.nElements() < 1:
                    print("warning, no data in file "+fullpath)
                del td
                newsamples.append(self.samples[i])
                if validsourcelist:
                    newsources.append(self.sourceList[i])
                continue
            except Exception as e:
                print('problem with file, removing ', fullpath)
                
        self.samples = newsamples
        self.newsources = newsources
                
    def removeEntry(self,relative_path_to_entry):
        for i in range(len(self.samples)):
            if relative_path_to_entry==self.samples[i]:
                print('removing '+self.samples[i])
                del self.samples[i]
                del self.sourceList[i]
                break
                 
        
    def writeToFile(self,filename):
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as fd:
            pickle.dump(self.samples, fd,protocol=0 )
            pickle.dump(self.sourceList, fd,protocol=0 )
            pickle.dump(self.dataclass, fd,protocol=0 )
            pickle.dump(self.weighterobjects, fd, protocol=0)
            pickle.dump(self.__batchsize, fd, protocol=0)
            pickle.dump(self.batch_uses_sum_of_squares, fd, protocol=0)
            pickle.dump(self.optionsdict, fd, protocol=0)
             
        shutil.move(fd.name, filename)
        
    def readFromFile(self,filename):
        fd=open(filename,'rb')
        self.samples=pickle.load(fd)
        self.sourceList=pickle.load(fd)
        try:
            self.dataclass=pickle.load(fd)
            self.weighterobjects=pickle.load(fd)
            self.__batchsize = pickle.load(fd)
            self.batch_uses_sum_of_squares = pickle.load(fd)
            self.optionsdict = pickle.load(fd)
        except Exception as e:
            print(e)
            print("WARNING: wrong dataCollection format. Can still be used for training(!), but it is advised to recreate it: this is possible without converting the original data again using the script createDataCollectionFromTD.py (takes a few seconds)")
        finally:
            fd.close()

        self.dataDir=os.path.dirname(os.path.abspath(filename))
        self.dataDir+='/'
        
        
    def readSourceListFromFile(self, file, relpath=''):
        self.samples=[]
        self.sourceList=[]
        self.__nsamples=0
        self.dataDir=""
        
        fdir=os.path.dirname(file)
        fdir=os.path.abspath(fdir)
        fdir=os.path.realpath(fdir)
        lines = [line.rstrip('\n') for line in open(file)]
        for line in lines:
            if len(line) < 1: continue
            if relpath:
                self.sourceList.append(os.path.join(relpath, line))
            else:
                self.sourceList.append(line)

        if len(self.sourceList)<1:
            raise Exception('source samples list empty')
        
        
    def split(self,ratio):
        '''
        out fraction is (1-ratio)
        returns out
        modifies self
        '''
        
        nin = int(len(self.samples)*(ratio))
        if nin < 1:
            raise ValueError("DataCollection:split: less than one sample would remain")
        
        if nin == len(self.samples):
            raise ValueError("DataCollection:split: less than one sample would be assigned to output")
            
        out=DataCollection()
        out.dataDir = self.dataDir
        out.dataclass = self.dataclass #anyway just a dummy
        out.samples = self.samples[nin:]
        self.samples = self.samples[:nin]
        
        if len(self.sourceList) == len(self.samples):
            out.sourceList = self.sourceList[nin:]
            self.sourceList = self.sourceList[:nin]
        else:
            self.sourceList = []
            out.sourceList = []
        
        #force re-read upon request
        self.__nsamples = 0
        out.__nsamples = 0
        
        out.weighterobjects = copy.deepcopy(self.weighterobjects)
        
        return out
        
    
    def recoverCreateDataFromRootFromSnapshot(self, snapshotfile):
        snapshotfile=os.path.abspath(snapshotfile)
        self.readFromFile(snapshotfile)

        if len(self.sourceList) < 1:
            return
        outputDir=os.path.dirname(snapshotfile)+'/'
        self.dataDir=outputDir
        finishedsamples=len(self.samples)
        
        self.__writeData_async_andCollect(finishedsamples,outputDir)
        self.writeToFile(outputDir+'/dataCollection.djcdc')
        
    
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
        
        if len(self.sourceList) < 1:
            print('createDataFromRoot: no input root file')
            raise Exception('createDataFromRoot: no input root file')
        
        outputDir+='/'
        if os.path.isdir(outputDir) and dir_check:
            raise Exception('output dir must not exist')
        elif not os.path.isdir(outputDir):
            os.mkdir(outputDir)
        self.dataDir=outputDir
        self.samples=[]
        self.dataclass=dataclass
        td=self.dataclass()

        self.weighterobjects = td.createWeighterObjects(self.sourceList)

        if self.batch_mode:
            for sample in self.sourceList:
                self.__writeData(sample, outputDir)
        else:
            self.__writeData_async_andCollect(0, outputDir)
    
    def __writeData(self, sample, outputDir):
        sw=stopwatch()
        td=self.dataclass()
        
        fileTimeOut(sample,120) #once available copy to ram

        sbasename = os.path.basename(sample)
        newname = sbasename[:sbasename.rfind('.')]+'.djctd'
        newpath=os.path.abspath(outputDir+newname)
        
        td.writeFromSourceFile(sample, self.weighterobjects, istraining=True, outname=newpath)
        
        print('converted and written '+newname+' in ',sw.getAndReset(),' sec')
        self.samples.append(newname)
        td.clear()
        
        if not self.batch_mode:
            self.writeToFile(outputDir+'/snapshot.djcdc')
            
        
    def __writeData_async_andCollect(self, startindex, outputDir):
        
        
        from multiprocessing import Process, Queue, cpu_count, Lock
        wo_queue = Queue()
        writelock=Lock()
        thispid=str(os.getpid())
        if not self.batch_mode and not os.path.isfile(outputDir+'/snapshot.djcdc'):
            self.writeToFile(outputDir+'/snapshot.djcdc')
        
        tempstoragepath='/dev/shm/'+thispid
        
        logger.info('creating dir '+tempstoragepath)
        os.system('mkdir -p '+tempstoragepath)
        
        def writeData_async(index,woq,wrlck):

            logger.info('async started')
            
            sw=stopwatch()
            td=self.dataclass()
            sample=self.sourceList[index]

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
            newname = sbasename[:sbasename.rfind('.')]+'.djctd'
            newpath=os.path.abspath(outputDir+newname)
            
            try:
                logger.info('convertFromSourceFile')
                td.writeFromSourceFile(tmpinput, self.weighterobjects, istraining = True, outname=newpath) 
                print('converted and written '+newname+' in ',sw.getAndReset(),' sec -', index)
                
                out_samplename=newname
                out_sampleentries=1
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
            if not self.batch_mode:
                self.writeToFile(outputDir+'/snapshot_tmp.djcdc')#avoid to overwrite directly
                os.system('mv '+outputDir+'/snapshot_tmp.djcdc '+outputDir+'/snapshot.djcdc')
            
        processes=[]
        processrunning=[]
        processfinished=[]
        for i in range(startindex,len(self.sourceList)):
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
                    logging.info('starting %s...' % self.sourceList[startindex+i])
                    processes[i].start()
                    processrunning[i]=True
                    nrunning+=1
                    
                
                
                if not wo_queue.empty():
                    res=wo_queue.get()
                    results.append(res)
                    originrootindex=res[0]
                    logging.info('finished %s...' % self.sourceList[originrootindex])
                    processfinished[originrootindex-startindex]=True
                    processes      [originrootindex-startindex].join(5)
                    processrunning [originrootindex-startindex]=False  
                    #immediately send the next
                    continue
                  
                results = sorted(results, key=lambda x:x[0])    
                for r in results:
                    thisidx=r[0]
                    if thisidx==lastindex+1:
                        logging.info('>>>> collected result %d of %d' % (thisidx+1,len(self.sourceList)))
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
            takeweightersfrom='', means_only=False,
            output_name='dataCollection.djcdc',
            relpath=''):
        
        newmeans=True
        if takeweightersfrom:
            self.readFromFile(takeweightersfrom)
            newmeans=False
        self.readSourceListFromFile(inputfile, relpath=relpath)
        self.createDataFromRoot(
                    dataclass, outputDir, 
                    newmeans, means_only = means_only, 
                    dir_check= not self.batch_mode
                    )
        self.writeToFile(outputDir+'/'+output_name)

    
    def getSamplePath(self,samplefile):
        #for backward compatibility
        if samplefile[0] == '/':
            return samplefile
        return self.dataDir+'/'+samplefile
    
    def __stackData(self, dataclass, selector):
        #
        # FIXME: use underlying trainData.append here (improvement)
        #
        td=self.dataclass()
        out=[]
        firstcall=True
        for sample in self.samples:
            td2 = self.dataclass()
            td2.readIn(self.getSamplePath(sample))
            td.append(td2)
        return td
    
    def invokeGenerator(self):
        self.generator = trainDataGenerator()
        self.generator.setBatchSize(self.__batchsize)
        self.generator.setSquaredElementsLimit(self.batch_uses_sum_of_squares)
        self.generator.setFileList([self.dataDir+ "/" + s for s in self.samples])
        
    
    def generatorFunction(self):
        
        
        while(1):
            ##needs to be adapted
            #
            # Use tf.ragged.from_rowsplits (or similar)
            #
            data = self.generator.getBatch()
            #print("batch size ", data.nElements())
            xout = data.transferFeatureListToNumpy()
            wout = data.transferWeightListToNumpy()
            yout = data.transferTruthListToNumpy()
            
            #debug
            #if self.generator.lastBatch(): # returns true if less than the previous batch size remains
            #    print('\nlastbatch')
            
            if len(wout)>0:
                yield (xout,yout,wout)
            else:
                yield (xout,yout)
            
        
    