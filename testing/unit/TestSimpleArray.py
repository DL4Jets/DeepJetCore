from DeepJetCore.SimpleArray import SimpleArray
import numpy as np
import unittest
import os

class TestSimpleArray(unittest.TestCase):


    def createNumpy(self,dtype):
        arr = np.array(np.random.rand(500,3,5,6)*100., dtype=dtype)
        rs = np.array([0,100,230,500],dtype='int64')
        return arr, rs

    def test_createFromNumpy(self):
        print('TestSimpleArray: createFromNumpy')
        arr,rs = self.createNumpy('float32')
        
        a = SimpleArray(dtype='float32')
        a.createFromNumpy(arr,rs)
        
        narr, nrs = a.copyToNumpy()

        diff = np.max(np.abs(narr-arr))
        diff += np.max(np.abs(nrs-rs))
        self.assertTrue(diff< 0.000001)
        
    def test_transferToNumpy(self):
        print('TestSimpleArray: transferToNumpy')
        arr,rs = self.createNumpy('float32')
        a = SimpleArray(arr,rs)
        narr, nrs = a.transferToNumpy()
        diff = np.max(np.abs(narr-arr))
        diff += np.max(np.abs(nrs-rs))
        self.assertTrue(diff< 0.000001)
        
        
    def test_transferToNumpyInt(self):
        print('TestSimpleArray: transferToNumpyInt')
        arr,rs = self.createNumpy('int32')
        a = SimpleArray(arr,rs)
        narr, nrs = a.transferToNumpy()
        diff = np.max(np.abs(narr-arr))
        diff += np.max(np.abs(nrs-rs))
        self.assertTrue(diff< 0.000001)
        
    def test_createFromNumpyInt(self):
        print('TestSimpleArray: createFromNumpyInt')
        
        arr,rs = self.createNumpy('int32')
        
        a = SimpleArray(dtype='int32')
        a.createFromNumpy(arr,rs)
        
        narr, nrs = a.copyToNumpy()
        diff = np.max(np.abs(narr-arr))
        diff += np.max(np.abs(nrs-rs))
        self.assertTrue(diff< 0.000001)
        
    def test_dynamicTypeChange(self):
        print('TestSimpleArray: dynamicTypeChange')
        arr,rs = self.createNumpy('int32')
        name = "lala"
        a = SimpleArray(dtype='float32',name=name)
        fnames = ["a","b","c","d","e","f"]
        a.setFeatureNames(fnames)
        a.createFromNumpy(arr,rs)
        self.assertTrue(a.featureNames() == fnames)
        self.assertTrue(a.name() == name)
        
        
    def test_writeRead(self):
        print('TestSimpleArray: writeRead')
        arr,rs = self.createNumpy('float32')
        
        a = SimpleArray(arr,rs)
        a.setName("myname")
        a.setFeatureNames(["a","b","c","d","e","f"])
        a.writeToFile("testfile.djcsa")
        b = SimpleArray()
        b.readFromFile("testfile.djcsa")
        os.system('rm -f testfile.djcsa')
        #os.system("rf -f testfile")
        
        ad, ars = a.copyToNumpy()
        bd, brs = b.copyToNumpy()
        diff = np.max(np.abs(ad-bd))
        diff += np.max(np.abs(ars-brs))
        self.assertTrue(diff==0)
        
            
    def test_equal(self):
        print('TestSimpleArray: equal')
        arr,rs = self.createNumpy('float32')
        
        a = SimpleArray()
        a.createFromNumpy(arr,rs)
        
        b = SimpleArray()
        b.createFromNumpy(arr,rs)
        
        self.assertEqual(a, b)
        
        b = a.copy()
        self.assertEqual(a, b)
        
        b.setFeatureNames(["a","b","c","d","e","f"])
        self.assertNotEqual(a, b)
        
        
        
        
    def test_append(self):
        print('TestSimpleArray: append')
        arr,rs = self.createNumpy('float32')
        
        arr2,_ = self.createNumpy('float32')
        
        a = SimpleArray(arr,rs)
        aa = SimpleArray(arr2,rs)
        a.append(aa)
        
        arr2 = np.concatenate([arr,arr2],axis=0)
        rs2 = rs.copy()[1:]
        rs2 += rs[-1]
        rs2 = np.concatenate([rs,rs2],axis=0)
        
        b = SimpleArray(arr2,rs2)
        self.assertEqual(a, b)
        
    def test_split(self):
        print('TestSimpleArray: split')
        
        arr,rs = self.createNumpy('float32')
        a = SimpleArray(arr,rs,name="myarray")
        
        arrs, rss = arr[:rs[2]], rs[:3]
        b = SimpleArray(arrs,rss,name="myarray")
        
        asplit = a.split(2)
        self.assertEqual(asplit, b)
        
    def test_name(self):
        print('TestSimpleArray: name')
        arr,rs=self.createNumpy('float32')
        a = SimpleArray(arr,rs)
        a.setName("myname")
        self.assertEqual("myname", a.name())
        
