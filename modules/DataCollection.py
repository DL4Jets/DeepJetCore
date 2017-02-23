'''
Created on 21 Feb 2017

@author: jkiesele
'''
#from tensorflow.contrib.labeled_tensor import batch

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
        self.useweights=True
        self.__batchsize=1
        
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
        #check if files exist
        for f in self.originRoots:
            if not os.path.isfile(f):
                print('not found: '+f)
                raise Exception('original root file not found')
        for f in self.samples:
            if not os.path.isfile(f):
                print('not found: '+f)
                raise Exception('sample file not found')
        
        
    def readRootListFromFile(self,file):
        import os
        self.clear()
        
        fdir=os.path.dirname(file)
        fdir=os.path.abspath(fdir)
        lines = [line.rstrip('\n') for line in open(file)]
        for line in lines:
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
           
        
        itself.useweights=self.useweights
        out.useweights=self.useweights
        
        
        itself.setBatchSize(self.__batchsize)
        out.setBatchSize(self.__batchsize)
        
        self=itself
        return out
    
    def createDataFromRoot(self,dataclass, outputDir):
        '''
        Also creates a file list of the output files
        After the operation, the object will point to the already processed
        files (not root files)
        '''
        import os
        import numpy
        outputDir+='/'
        if os.path.isdir(outputDir):
            raise Exception('output dir must not exist')
        os.mkdir(outputDir)
        self.nsamples=0
        self.samples=[]
        self.sampleentries=[]
        means=numpy.array([])
        firstSample=True
        for sample in self.originRoots:
            td=dataclass
            print ('creating '+ str(type(dataclass)) +' data from '+sample)
            os.path.abspath(sample)
            
            if firstSample:
                print('producting means')
                means=td.produceMeansFromRootFile(sample)
                firstSample=False
                
            td.readFromRootFile(sample,means) 
            newname=os.path.basename(sample).rsplit('.', 1)[0]
            newpath=os.path.abspath(outputDir+newname+'.z')
            td.writeOut(newpath)
            self.samples.append(newpath)
            self.nsamples+=td.nsamples
            self.sampleentries.append(td.nsamples)
            td.clear()
        
        
    def convertListOfRootFiles(self, inputfile, dataclass, outputDir):
        self.readRootListFromFile(inputfile)
        self.createDataFromRoot(dataclass, outputDir)
        self.writeToFile(outputDir+'/dataCollection.dc')
        
    def getOneFileLabels(self, dataclass):
        td=dataclass
        td.readIn(self.samples[0])
        return td.y
        
    def getOneFileFeatures(self, dataclass):
        td=dataclass
        td.readIn(self.samples[0])
        return td.x
        
    def getOneFileWeights(self, dataclass):
        td=dataclass
        td.readIn(self.samples[0])
        return td.w
        
        
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
            if lastbatchrest > self.__batchsize:
                batchcomplete = True
                
            while not batchcomplete:
                readsample=self.samples[nextfiletoread]
                td.readIn(readsample)
                #get the format right in the first read
                if nextfiletoread == 0:
                    
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
                    
                if xstored[0].shape[0] > self.__batchsize:
                    batchcomplete = True
                        
                td.clear()
                
                nextfiletoread+=1
                if nextfiletoread >= len(self.samples):
                    nextfiletoread=0
                
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
            
            

    
    
    
    