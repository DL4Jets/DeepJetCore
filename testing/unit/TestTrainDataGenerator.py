import numpy as np
from DeepJetCore import TrainData, DataCollection
import shutil
import unittest

class RaggedTester(object):
    def __init__(self, max_per_rs=543):
        self.max_per_rs=max_per_rs

    def createEvent(self,length: int,dtype='float32'):
        a = np.arange(length,dtype=dtype)
        a = np.expand_dims(a,axis=1)
        return a
    
    def checkEvent(self,a,dtype='float32'):
        checkarr = self.createEvent(len(a),dtype)
        return np.all(checkarr==a) and checkarr.dtype == a.dtype
        
    
    
    def createData(self,ntotal):
        segments = np.random.randint(2,self.max_per_rs,size=ntotal) #8347
        row_splits = [0]
        data=[]
        for s in segments:
            data.append(self.createEvent(s))
            row_splits.append(s+row_splits[-1])
        
        return np.concatenate(data), np.array(row_splits,dtype='int64')

    def checkData(self,data,rs,dtype='float32'):
        for i in range(len(rs)-1):
            ea=data[rs[i]:rs[i+1]]
            if not self.checkEvent(ea,dtype):
                return False
        return True
 
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
        from DeepJetCore import SimpleArray
        
        seed = int(hashlib.sha1(filename.encode('utf-8')).hexdigest(), 16) % (10 ** 8)
        np.random.seed(seed)
        nsamples = np.random.randint(12,101,size=1)
        data,rs = raggedtester.createData(nsamples)
        
        farr = SimpleArray(data, rs,name="features_ragged")
        true_arr = SimpleArray(data, rs,name="truth_ragged")
        farrint = SimpleArray(np.array(data,dtype='int32'), rs, name="features_int_ragged")
        #farr.createFromNumpy()
        
        return [farr,farrint],[true_arr],[]
    


class TestTrainDataGenerator(unittest.TestCase):
        
    def test_fullGenerator(self):
        print("TestTrainDataGenerator full generator")
        
        passed = True
        
        n_files=11
        n_per_batch=2078
        files = TempFileList(n_files)
        dcoutdir = TempDirName()
    
        n_per_batch=n_per_batch
        
        dc = DataCollection()
        dc.dataclass = TrainData_test
        dc.sourceList = [f for f in files.filenames]
        dc.createDataFromRoot(TrainData_test, outputDir=dcoutdir.path)
        
        gen = dc.invokeGenerator()
        gen.setBatchSize(n_per_batch)
        
        for epoch in range(10):
            gen.prepareNextEpoch()
            for b in range(gen.getNBatches()):
                d,t = next(gen.feedNumpyData())
                data,rs, dint, _ = d[0],d[1],d[2],d[3]
                truth = t[0]
                rs = rs[:,0]#remove last 1 dim
                
                datagood = raggedtester.checkData(data, rs)
                datagood = datagood and raggedtester.checkData(dint, rs, 'int32')
                datagood = datagood and raggedtester.checkData(truth, rs)
                
                if not datagood:
                    print('epoch',epoch, 'batch',b,'broken')
                    passed=False
                    break
                if rs[-1] > n_per_batch:
                    print('maximum batch size exceeded for batch ',b, 'epoch', epoch)
                    passed = False
                    break
                
            gen.shuffleFileList()
            
        shutil.rmtree(dcoutdir.path)
        self.assertTrue(passed)
        
    def test_fullGeneratorDict(self):
        print("TestTrainDataGenerator full generator with dictionary")
        
        passed = True
        
        n_files=11
        n_per_batch=2078
        files = TempFileList(n_files)
        dcoutdir = TempDirName()
    
        n_per_batch=n_per_batch
        
        dc = DataCollection()
        dc.dataclass = TrainData_test
        dc.sourceList = [f for f in files.filenames]
        dc.createDataFromRoot(TrainData_test, outputDir=dcoutdir.path)
        
        gen = dc.invokeGenerator()
        gen.setBatchSize(n_per_batch)
        gen.dict_output = True
        
        for epoch in range(10):
            gen.prepareNextEpoch()
            for b in range(gen.getNBatches()):
                d,t = next(gen.feedNumpyData())
                data,rs, dint = d['features_ragged'],d['features_ragged_rowsplits'],d['features_int_ragged']
                truth = t['truth_ragged']
                rs = rs[:,0]#remove last 1 dim
                
                datagood = raggedtester.checkData(data, rs)
                datagood = datagood and raggedtester.checkData(dint, rs, 'int32')
                datagood = datagood and raggedtester.checkData(truth, rs)
                
                if not datagood:
                    print('epoch',epoch, 'batch',b,'broken')
                    passed=False
                    break
                if rs[-1] > n_per_batch:
                    print('maximum batch size exceeded for batch ',b, 'epoch', epoch)
                    passed = False
                    break
                
            gen.shuffleFileList()
            
        shutil.rmtree(dcoutdir.path)
        self.assertTrue(passed)
