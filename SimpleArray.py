 
from DeepJetCore.compiled.c_simpleArray import simpleArrayF, simpleArrayI
import numpy as np

class SimpleArray(object):
    
    def __init__(self, nparr=None, nprs=np.array([],dtype='int64'), dtype='float32', name=""):
        
        assert nparr is not None or dtype is not None
        self.dtype=None
        if nparr is not None:
            dtype = str(nparr.dtype)
        self._setDtype(dtype)
        if nparr is not None:
            self.createFromNumpy(nparr, nprs)
        self.setName(name)
        
    def __eq__(self,other):
        if self.sa.dtypeI() != other.sa.dtypeI():
            return False
        return self.sa == other.sa
            
    def _setDtype(self,dtype):
        assert dtype=='float32' or dtype=='int32'
        if dtype=='float32':
            self.sa = simpleArrayF()
        elif dtype=='int32':
            self.sa = simpleArrayI()
        self.dtype = dtype
            
    #now pass through all the other member functions transparently
    
    def setName(self, namestr: str):
        self.sa.setName(namestr)
        
    def setFeatureNames(self, names: list):
        self.sa.setFeatureNames(names)

    def name(self):
        return self.sa.name()
    
    def featureNames(self):
        return self.sa.featureNames()

    def readFromFile(self,filename):
        dt = self.sa.readDtypeFromFile(filename)
        self._setDtype(dt)
        return self.sa.readFromFile(filename)
    
    def writeToFile(self,*args):
        return self.sa.writeToFile(*args) 
        
    def assignFromNumpy(self,*args):
        return self.sa.assignFromNumpy(*args)
        
    def createFromNumpy(self, nparr, nprs=np.array([],dtype='int64')):
        if nprs.dtype == 'int32':
            return self.sa.createFromNumpy(nparr, nprs.as_type('int64')) 
        return self.sa.createFromNumpy(nparr, nprs) 
        
    
    def copyToNumpy(self, pad_rowsplits=False):
        return self.sa.copyToNumpy(pad_rowsplits) 
    
    def transferToNumpy(self, pad_rowsplits=False):
        return self.sa.transferToNumpy(pad_rowsplits) 
    
    def isRagged(self,*args):
        return self.sa.isRagged(*args) 
    
    def split(self,*args):
        spl = SimpleArray()
        spl._setDtype(self.dtype)
        spl.sa = self.sa.split(*args)
        return spl
    
    def getSlice(self,*args):
        return self.sa.getSlice(*args)
    
    def append(self,other):
        assert self.sa.dtypeI() == other.sa.dtypeI()
        return self.sa.append(other.sa)
    
    def cout(self,*args):
        return self.sa.cout(*args)
    
    def size(self,*args):
        return self.sa.size(*args)
    
    def copy(self):
        arr,rs = self.sa.copyToNumpy(False)
        return SimpleArray(arr,rs)
    
