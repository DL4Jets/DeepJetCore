'''
Checks for file compatibility with (only) the previous version.
'''

from DeepJetCore.TrainData import TrainData
from DeepJetCore.SimpleArray import SimpleArray
import numpy as np
import unittest


class TestCompatibility(unittest.TestCase):
    
    def test_SimpleArrayRead(self):
        print('TestCompatibility SimpleArray')
        a = SimpleArray()
        a.readFromFile("simpleArray_previous.djcsa")
        
        arr = np.load("np_arr.npy")
        #FIXME: this array was actually wrong
        arr = arr[:100]
        rs = np.load("np_rs.npy")
        
        b = SimpleArray(arr,rs)
        
        self.assertEqual(a,b)
        
    def test_TrainDataRead(self):
        print('TestCompatibility TrainData')
        td = TrainData()
        td.readFromFile('trainData_previous.djctd')
        
        self.assertEqual(td.nFeatureArrays(), 1)
        
        arr = np.load("np_arr.npy")
        #FIXME: this array was actually wrong
        arr = arr[:100]
        rs = np.load("np_rs.npy")
        
        b = SimpleArray(arr,rs)
        
        a = td.transferFeatureListToNumpy(False)
        a, rs = a[0],a[1]
        
        a = SimpleArray(a,np.array(rs,dtype='int64'))
        
        self.assertEqual(a,b)
        