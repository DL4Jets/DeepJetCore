
from DeepJetCore.compiled.c_trainDataGenerator import trainDataGenerator
import numpy as np

class TrainDataGenerator(trainDataGenerator):
    
    def __init__(self, 
                 pad_rowsplits=False, 
                 fake_truth=None,
                 dict_output=False,
                 cast_to = None):
        
        trainDataGenerator.__init__(self)
        #self.extend_truth_list_by = extend_truth_list_by
        self.pad_rowsplits=pad_rowsplits
        self.dict_output = dict_output
        self.fake_truth = None
        self.cast_to = cast_to
        if fake_truth is not None:
            if isinstance(fake_truth, int):
                self.fake_truth = [np.array([0],dtype='float32') 
                                             for _ in range(fake_truth)]
            elif isinstance(fake_truth, list):
                etl={}
                for e in fake_truth:
                    if isinstance(e,str):
                        etl[e]=np.array([0],dtype='float32') 
                    else:
                        raise ValueError("TrainDataGenerator: only accepts an int or list of strings to extend truth list")
                self.fake_truth = etl
    
    def feedTrainData(self):
        for _ in range(self.getNBatches()):
            td = self.getBatch()
            if self.cast_to is not None:
                td.__class__ = self.cast_to
            yield td
        
    def feedNumpyData(self):
        
        fnames=[]
        tnames=[]
        wnames=[]
        
        for b in range(self.getNBatches()):
            try:
                data = self.getBatch()
                
                if not len(fnames):
                    fnames = data.getNumpyFeatureArrayNames()
                    tnames = data.getNumpyTruthArrayNames()
                    wnames = data.getNumpyWeightArrayNames()
                
                # These calls will transfer data to numpy and delete the respective SimpleArray
                # instances for efficiency.
                # therefore extracting names etc needs to happen before!
                xout = data.transferFeatureListToNumpy(self.pad_rowsplits)
                yout = data.transferTruthListToNumpy(self.pad_rowsplits)
                wout = data.transferWeightListToNumpy(self.pad_rowsplits)
                
                if self.dict_output:
                    xout = {k:v for k,v in zip(fnames,xout)}
                    yout = {k:v for k,v in zip(tnames,yout)}
                    wout = {k:v for k,v in zip(wnames,wout)}
                
                if self.fake_truth is not None:
                    yout=self.fake_truth
                
                out = (xout,yout)
                if len(wout)>0:
                    out = (xout,yout,wout)
                yield out
            except Exception as e:
                print("TrainDataGenerator: an exception was raised in batch",b," out of ", self.getNBatches(),', expection: ', e)
                raise e
            
    def feedTorchTensors(self):
        pass