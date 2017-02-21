'''
Created on 21 Feb 2017

@author: jkiesele
'''

#for convenience


class DataCollection(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.samples=[]
        
        
    def readFromFile(self,file):
        import os
        fdir=os.path.dirname(file)
        fdir=os.path.abspath(fdir)
        lines = [line.rstrip('\n') for line in open(file)]
        for line in lines:
            self.samples.append(fdir+'/'+line)
    
        if len(self.samples)<1:
            raise Exception('samples list empty')
        
    def splitToTrainAndTest(self,ratio):
        '''
        returns the training and test object
        '''
        
        trainsamples=[]
        testsamples=[]
        
        rcount=0
        nlines=len(self.samples)
        
        for line in self.samples:
            if rcount/nlines < ratio and rcount < nlines-1:
                trainsamples.append(line)
            else:
                testsamples.append(line)
            rcount+=1
        
        
        train=DataCollection()
        train.samples=trainsamples
        test=DataCollection()
        test.samples=testsamples
        
        return [train,test]
    
    def createDataFromRoot(self,dataclass, outputDir):
        '''
        Also creates a file list of the output files
        After the operation, the object will point to the already processed
        files (not root files)
        '''
        import os
        outputDir+='/'
        if os.path.isdir(outputDir):
            raise Exception('output dir must not exist')
        os.mkdir(outputDir)
        
        newlist=[]
        for sample in self.samples:
            td=dataclass
            td.readFromRootFile(sample)
            newname=os.path.basename(sample).rsplit('.', 1)[0]
            td.writeOut(outputDir+newname)
            newlist.append(newname)
            td.clear()
        newfile = open(outputDir+'samples.txt', 'w')
        for item in newlist:
            newfile.write("%s\n" % item)
        newfile.close()
        
        self.samples=newlist
        
        
        
    def convertListOfRootFiles(self, inputfile, dataclass, outputDir):
        self.readFromFile(inputfile)
        self.createDataFromRoot(dataclass, outputDir)
        
    def generator(self, dataclass):
        while 1:
            for sample in self.samples:
                td=dataclass
                td.readIn(sample)
                yield (td.x,td.y,td.w)
                td.clear()

    
    
    
    