from DeepJetCore.TrainData import TrainData
from DeepJetCore.SimpleArray import SimpleArray
import numpy as np
import unittest
import os

class TestTrainData(unittest.TestCase):
    
    def createSimpleArray(self, dtype, length=500, shape=None):
        arr = np.array(np.random.rand(length,3,5,6)*100., dtype=dtype)
        rs = np.array([0,100,230,length],dtype='int64')
        return SimpleArray(arr, rs)
    
    def sub_test_store(self, readWrite):
        td = TrainData()
        x,y,w = self.createSimpleArray('int32'), self.createSimpleArray('float32'), self.createSimpleArray('int32')
        x_orig=x.copy()
        x2,y2,_ = self.createSimpleArray('float32'), self.createSimpleArray('float32'), self.createSimpleArray('int32')
        x2_orig=x2.copy()
        y_orig=y.copy()
        
        td._store([x,x2], [y,y2], [w])
        
        if readWrite:
            td.writeToFile("testfile.tdjctd")
            td = TrainData()
            td.readFromFile("testfile.tdjctd")
            os.system('rm -f testfile.tdjctd')
        
        shapes = td.getNumpyFeatureShapes()
        self.assertEqual([[3, 5, 6], [1], [3, 5, 6], [1]], shapes,"shapes")
        
        self.assertEqual(2, td.nFeatureArrays())
        self.assertEqual(2, td.nTruthArrays())
        self.assertEqual(1, td.nWeightArrays())
        
        f = td.transferFeatureListToNumpy(False)
        t = td.transferTruthListToNumpy(False)
        w = td.transferWeightListToNumpy(False)
        
        xnew = SimpleArray(f[0],np.array(f[1],dtype='int64'))
        self.assertEqual(x_orig, xnew)
        
        xnew = SimpleArray(f[2],np.array(f[3],dtype='int64'))
        self.assertEqual(x2_orig, xnew)
        
        ynew = SimpleArray(t[0],np.array(t[1],dtype='int64'))
        self.assertEqual(y_orig, ynew)
        
    def test_store(self):  
        print('TestTrainData: store')
        self.sub_test_store(False)  
        
    def test_readWrite(self):
        print('TestTrainData: readWrite')
        self.sub_test_store(True) 
        
    def nestedEqual(self,l,l2):
        for a,b in zip(l,l2):
            if not np.all(a==b):
                return False
        return True
    
    def test_AddToFile(self):
        print('TestTrainData: AddToFile')
        
        td = TrainData()
        x,y,w = self.createSimpleArray('int32'), self.createSimpleArray('float32'), self.createSimpleArray('int32')
        xo,yo,wo = x.copy(),y.copy(),w.copy()
        x2,y2,_ = self.createSimpleArray('float32'), self.createSimpleArray('float32'), self.createSimpleArray('int32')
        x2o,y2o = x2.copy(),y2.copy()
        td._store([x,x2], [y,y2], [w])
        
        td.writeToFile("testfile.tdjctd")
        td.addToFile("testfile.tdjctd")
        
        
        td2 = TrainData()
        td2._store([xo,x2o], [yo,y2o], [wo])
        td2.append(td)
        
        td.readFromFile("testfile.tdjctd")
        os.system('rm -f testfile.tdjctd')
        
        
        self.assertEqual(td,td2)
        
    def test_slice(self):
        print('TestTrainData: skim')
        a = self.createSimpleArray('int32',600)
        b = self.createSimpleArray('float32',600)
        d = self.createSimpleArray('float32',600)

        a_slice = a.getSlice(2,3)
        b_slice = b.getSlice(2,3)
        d_slice = d.getSlice(2,3)

        td = TrainData()
        td._store([a,b], [d], [])
        td_slice = td.getSlice(2,3)
        
        fl = td_slice.transferFeatureListToNumpy(False)
        tl = td_slice.transferTruthListToNumpy(False)
        a_tdslice = SimpleArray(fl[0],fl[1])
        b_tdslice = SimpleArray(fl[2],fl[3])
        d_tdslice = SimpleArray(tl[0],tl[1])

        self.assertEqual(a_slice, a_tdslice)
        self.assertEqual(b_slice, b_tdslice)
        self.assertEqual(d_slice, d_tdslice)
        
        #test skim
        td.skim(2)
        fl = td.transferFeatureListToNumpy(False)
        tl = td.transferTruthListToNumpy(False)
        a_tdslice = SimpleArray(fl[0],fl[1])
        b_tdslice = SimpleArray(fl[2],fl[3])
        d_tdslice = SimpleArray(tl[0],tl[1])
        
        self.assertEqual(a_slice, a_tdslice)
        self.assertEqual(b_slice, b_tdslice)
        self.assertEqual(d_slice, d_tdslice)
        
            
        
    def test_split(self):
        print('TestTrainData: split')
        a = self.createSimpleArray('int32')
        b = self.createSimpleArray('float32',600)
        c = self.createSimpleArray('int32')
        d = self.createSimpleArray('float32',400)
        all_orig = [a.copy(),b.copy(),c.copy(),d.copy()]
        all_splitorig = [sa.split(2) for sa in all_orig]
        
        td = TrainData()
        td._store([a,b], [c,d], [])
        
        
        tdb = td.split(2)
        f = tdb.transferFeatureListToNumpy(False)
        t = tdb.transferTruthListToNumpy(False)
        _ = tdb.transferWeightListToNumpy(False)
        all_split = [SimpleArray(f[0],f[1]), SimpleArray(f[2],f[3]),
                     SimpleArray(t[0],t[1]), SimpleArray(t[2],t[3])]
        
        self.assertEqual(all_splitorig,all_split)
        
    def test_KerasDTypes(self):
        print('TestTrainData: split')
        a = self.createSimpleArray('int32')
        b = self.createSimpleArray('float32',600)
        c = self.createSimpleArray('int32')
        d = self.createSimpleArray('float32',400)
        
        td = TrainData()
        td._store([a,b], [c,d], [])
        
        #data, rs, data, rs
        self.assertEqual(td.getNumpyFeatureDTypes(), ['int32','int64','float32','int64'])
        