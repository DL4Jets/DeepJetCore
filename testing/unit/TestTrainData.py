from DeepJetCore.TrainData import TrainData
from DeepJetCore.SimpleArray import SimpleArray
import numpy as np
import unittest
import os

class TestTrainData(unittest.TestCase):
    
    def createSimpleArray(self, dtype):
        arr = np.array(np.random.rand(500,3,5,6)*100., dtype=dtype)
        rs = np.array([0,100,230,500],dtype='int64')
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
        
        shapes = td.getKerasFeatureShapes()
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
        
        
        
    
        