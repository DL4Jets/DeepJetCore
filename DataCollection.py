'''
Created on 21 Feb 2017

@author: jkiesele
'''


class DataCollection(object):
    '''
    classdocs
    '''


    def __init__(self, infile = None, nprocs = -1):
        '''
        Constructor
        '''
        self.clear()
        if infile:
            self.readFromFile(infile)
            if not len(self.samples):
                raise Exception("no valid datacollection found in "+infile)
            
        self.weighterobjects=()

        
    def clear(self):
        self.samples=[]
        self.sampleentries=[]
        self.sourceList=[]
        self.nsamples=0
        self.dataDir=""
        self.dataclass=TrainData()
        

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
        _extend_(self, other, 'sourceList')
        self.nsamples += other.nsamples
        if self.dataDir != other.dataDir:
            raise ValueError('The two DataCollections have different data directories, still to be implemented!')
        if type(self.dataclass) != type(other.dataclass):
            raise ValueError(
                'The two DataCollections were made with a'
                ' different data class type! (%s, and %s)' % (type(self.dataclass), type(other.dataclass))
                )
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
        self.sourceList.pop()
        
       
    def getInputShapes(self):
        '''
        gets the input shapes from the data class description
        '''
        if len(self.samples)<1:
            return []
        self.dataclass.filelock=None
        td=copy.deepcopy(self.dataclass)
        td.readFromFile(self.getSamplePath(self.samples[0]),shapesOnly=True)
        shapes=td.getInputShapes()
        td.clear()
        return shapes
    
    def getTruthShape(self):
        return self.dataclass.getTruthShapes()
        
    def setBatchSize(self,bsize):
        if bsize > self.nsamples:
            raise Exception('Batch size must not be bigger than total sample size')
        self.__batchsize=bsize

    def getBatchSize(self):
        return self.__batchsize
    
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
                td.readFromFile(fullpath)
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
                del self.sourceList[i]
                self.nsamples -= self.sampleentries[i]
                del self.sampleentries[i]
                
    def removeEntry(self,relative_path_to_entry):
        for i in range(len(self.samples)):
            if relative_path_to_entry==self.samples[i]:
                print('removing '+self.samples[i]+" - "+str(self.sampleentries[i]))
                del self.samples[i]
                del self.sourceList[i]
                self.nsamples -= self.sampleentries[i]
                del self.sampleentries[i]
                break
                 
        
    def writeToFile(self,filename):
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as fd:
            self.dataclass.clear()
            pickle.dump(self.samples, fd,protocol=0 )
            pickle.dump(self.sampleentries, fd,protocol=0 )
            pickle.dump(self.sourceList, fd,protocol=0 )
            pickle.dump(self.nsamples, fd,protocol=0 )
            pickle.dump(self.dataclass, fd,protocol=0 )
            pickle.dump(self.self.weighterobjects, fd, protocol=0)
            self.means.dump(fd)

        shutil.move(fd.name, filename)
        
    def readFromFile(self,filename):
        fd=open(filename,'rb')
        self.samples=pickle.load(fd)
        self.sampleentries=pickle.load(fd)
        self.sourceList=pickle.load(fd)
        self.nsamples=pickle.load(fd)
        self.dataclass=pickle.load(fd)
        self.weighterobjects=pickle.load(fd)
        fd.close()

        self.dataDir=os.path.dirname(os.path.abspath(filename))
        self.dataDir+='/'
        
        
    def readSourceListFromFile(self, file, relpath=''):
        self.samples=[]
        self.sampleentries=[]
        self.sourceList=[]
        self.nsamples=0
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
        out.sourceList=[]
        out.nsamples=0
        out.__batchsize=copy.deepcopy(self.__batchsize)
        out.dataDir=self.dataDir

        out.dataclass=copy.deepcopy(self.dataclass)
        
        itself.samples=[]
        itself.sampleentries=[]
        itself.sourceList=[]
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
                itself.sourceList.append(self.sourceList[i])
                itself.nsamples+=self.sampleentries[i]
            else:
                out.samples.append(self.samples[i])
                out.sampleentries.append(self.sampleentries[i])
                out.sourceList.append(self.sourceList[i])
                out.nsamples+=self.sampleentries[i]
           
        
        self.samples=itself.samples
        self.sampleentries=itself.sampleentries
        self.sourceList=itself.sourceList
        self.nsamples=itself.nsamples
        
        return out
        
    
    def recoverCreateDataFromRootFromSnapshot(self, snapshotfile):
        snapshotfile=os.path.abspath(snapshotfile)
        self.readFromFile(snapshotfile)
        td=self.dataclass

        if len(self.sourceList) < 1:
            return
        #if not self.means:
        #    self.means=td.produceMeansFromRootFile(self.sourceList[0])
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
        
        if len(self.sourceList) < 1:
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

        self.weighterobjects = td.createWeighterObjects(self.sourceList)

        if self.batch_mode:
            for sample in self.sourceList:
                self.__writeData(sample, outputDir)
        else:
            self.__writeData_async_andCollect(0, outputDir)
    
    def __writeData(self, sample, outputDir):
        sw=stopwatch()
        td=copy.deepcopy(self.dataclass)
        
        fileTimeOut(sample,120) #once available copy to ram

        td.convertFromSourceFile(sample, self.weighterobjects)
        
        sbasename = os.path.basename(sample)
        newname = sbasename[:sbasename.rfind('.')]+'.djctd'
        
        newpath=os.path.abspath(outputDir+newname)
        td.writeToFile(newpath)
        print('converted and written '+newname+' in ',sw.getAndReset(),' sec')
        self.samples.append(newname)
        self.nsamples+=td.nsamples
        self.sampleentries.append(td.nsamples)
        td.clear()
        
        if not self.batch_mode:
            self.writeToFile(outputDir+'/snapshot.dc')
            
        
    def __writeData_async_andCollect(self, startindex, outputDir):
        
        
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
            newname = sbasename[:sbasename.rfind('.')]+'djctd'
            newpath=os.path.abspath(outputDir+newname)
            
            try:
                logger.info('readFromSourceFile')
                td.readFromSourceFile(tmpinput, self.weighterobjects)
                logger.info('writeOut')
                #wrlck.acquire()
                td.writeToFile(newpath)
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
            output_name='dataCollection.dc',
            relpath=''):
        
        newmeans=True
        if takemeansfrom:
            self.readFromFile(takeweightersfrom)
            newmeans=False
        self.readSourceListFromFile(inputfile, relpath=relpath)
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
    
        
    def generator(self):
        from DeepJetCore.compiled.c_dataGenerator import numpyGenerator
        
        gen = numpyGenerator()
        gen.setFileList(self.samples)
        gen.setBatchSize(self.__batchsize)
        gen.setNTotal(self.nsamples)
        gen.beginEpoch()
        
        while(1):
            
            data = gen.getBatch(0)
            xout = data[0]
            yout = data[1]
            wout = data[2]
            
            if gen.lastBatch() :
                gen.endEpoch()
            
            if len(wout)>0:
                yield (xout,yout,wout)
            else:
                yield (xout,yout)
            
            

    
    
    
    
