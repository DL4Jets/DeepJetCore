

from TrainData import TrainData, fileTimeOut
import numpy


class TrainData_test(TrainData):
    '''
    class to make tests.
    Generates random data regardless of the input files
    Can be useful for testing technical things
    '''
    
    def __init__(self):
        import numpy
        TrainData.__init__(self)
        self.remove=False
            
        #define truth:
        self.undefTruth=['']
    
        self.truthclasses=['isClassA',
                            'isClassB']
        
        self.regressiontargetclasses=['reg']
        
        self.reduceTruth()
       
    def produceBinWeighter(self,filenames):
        return self.make_empty_weighter()   
    
    def produceMeansFromRootFile(self,files,limit):
        return numpy.array([1])
        
    def reduceTruth(self, tuple_in=None):
        self.reducedtruthclasses=['isClassA',
                            'isClassB']  
        
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        
        import hashlib
        m = hashlib.md5()
        seed=int(int(m.hexdigest(),16)/1e36)
        
        
        
        numpy.random.seed(seed)
        ya=numpy.random.random_integers(0,1,10000)
        yb=numpy.zeros(10000)
        yb=yb+1
        yb=yb-ya
        y=numpy.vstack((ya,yb))
        yclass=y.transpose()
        
        numpy.random.seed(seed)
        xreg=numpy.random.randn(10000)
        numpy.random.seed(seed)
        y=numpy.random.randn(10000)/2
        y=y+1
        w=numpy.zeros(10000)
        w=w+1
        
        xclass=y+ya

        print(xclass.shape)
        print(xreg.shape)
        print(yclass.shape)
        print(y.shape)
        
        self.nsamples=10000
        
        self.w=[w,w]
        self.x=[xclass,xreg]
        self.y=[yclass,y]
        
        
        
        