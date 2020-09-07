import numpy as np
from DeepJetCore.TrainData import TrainData
from DeepJetCore.DataCollection import DataCollection
import shutil

class RaggedTester(object):
    def __init__(self, auto_create=0):
        self.max_per_rs=3478
        self.fill_freq=11
        self.fill_content=11
        self.data=np.zeros((0,1),dtype='float32')
        self.rs=np.array([],dtype='int64')
        if auto_create > 0:
            self.data,self.rs = self.createData(auto_create)

    def createEvent(self,length: int):
        a = np.zeros((length,1),dtype='float32')
        for i in range(length//self.fill_freq - 1):
            a[(i+1)*self.fill_freq,0] = self.fill_content
        a[0]=1000
        a[-1]=2000
        return a
    
    def checkEvent(self,a):
        boundary = a[0,0] == 1000 and a[-1,0] == 2000
        internal=True
        for i in range(len(a)//self.fill_freq -1):
            if not a[(i+1)*self.fill_freq,0] == self.fill_content:
                internal = False
        return boundary and internal
    
    
    def createData(self,ntotal):
        segments = np.random.randint(2,self.max_per_rs,size=ntotal) #8347
        row_splits = [0]
        data=[]
        for s in segments:
            data.append(self.createEvent(s))
            row_splits.append(s+row_splits[-1])
        
        return np.concatenate(data), np.array(row_splits,dtype='int64')

    def checkData(self,data,rs):
        inputOk=True
        for i in range(len(rs)-1):
            ea=data[rs[i]:rs[i+1]]
            inputOk = inputOk and self.checkEvent(ea)
        return inputOk
 
import tempfile 
class TempFileList(object):
    def __init__(self, length=1):
        self._files=[tempfile.NamedTemporaryFile(delete=True) for _ in range(length)]
        self.filenames = [f.name for f in self._files]
        
    def __del__(self):
        for f in self._files:
            f.close()
    
        
            
class TempDir(object):
    def __init__(self, delete=True):
        self.path = tempfile.mkdtemp()
        self.delete=delete
        print(self.path)
        
    def __del__(self):
        if not self.delete:
            return
        shutil.rmtree(self.path)
        
class TempDirName(object):
    def __init__(self):
        td=TempDir(delete=True)
        self.path=td.path
        del td
    
          

## self-consistency check

raggedtester=RaggedTester()

class TrainData_test(TrainData):
    
    def convertFromSourceFile(self, filename, weighterobjects, istraining):
        global raggedtester
        import hashlib      
        from DeepJetCore.compiled.c_simpleArray import simpleArray
        
        seed = int(hashlib.sha1(filename.encode('utf-8')).hexdigest(), 16) % (10 ** 8)
        np.random.seed(seed)
        nsamples = np.random.randint(12,5415,size=1)
        data,rs = raggedtester.createData(nsamples)
        
        farr = simpleArray()
        farr.createFromNumpy(data, rs)
        
        return [farr],[],[]
    


class RaggedGeneratorTester(object):
    def __init__(self,
                 n_files=20,
                 n_per_batch=2078,
                 ):
        
        self.files = TempFileList(n_files)
        self.dcoutdir = TempDirName()
        self.n_per_batch=n_per_batch
        
    def __del__(self):
        shutil.rmtree(self.dcoutdir.path)
        
    
    def test(self):
        
        passed = True
        
        dc = DataCollection()
        dc.dataclass = TrainData_test
        dc.sourceList = [f for f in self.files.filenames]
        dc.createDataFromRoot(TrainData_test, outputDir=self.dcoutdir.path)
        
        gen = dc.invokeGenerator()
        gen.setBatchSize(self.n_per_batch)
        
        for epoch in range(10):
            gen.prepareNextEpoch()
            print("epoch",epoch,'batches',gen.getNBatches())
            for b in range(gen.getNBatches()):
                d,_ = next(gen.feedNumpyData())
                data,rs = d[0],d[1]
                rs = np.array(rs[:,0],dtype='int')
                rs = rs[:rs[-1]]
                if not raggedtester.checkData(data, rs):
                    print('epoch',epoch, 'batch',b,'broken')
                    passed=False
                    break
            print('shuffling')
            gen.shuffleFilelist()
            
        return passed
        
        
tester = RaggedGeneratorTester()
tester.test()






