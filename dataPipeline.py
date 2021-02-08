
from DeepJetCore.compiled.c_trainDataGenerator import trainDataGenerator
import numpy as np

class TrainDataGenerator(trainDataGenerator):
    
    def __init__(self, pad_rowsplits=False, extend_truth_list_by=0):
        trainDataGenerator.__init__(self)
        self.extend_truth_list_by = extend_truth_list_by
        self.pad_rowsplits=pad_rowsplits
        
    def feedNumpyData(self):
        
        for b in range(self.getNBatches()):
            try:
                data = self.getBatch()
                
                xout = data.transferFeatureListToNumpy(self.pad_rowsplits)
                wout = data.transferWeightListToNumpy(self.pad_rowsplits)
                yout = data.transferTruthListToNumpy(self.pad_rowsplits)
                
                if self.extend_truth_list_by > 0:
                    tadd = [np.array([0],dtype='float32') for _ in range(self.extend_truth_list_by)]
                    yout += tadd
                
                out = (xout,yout)
                if len(wout)>0:
                    out = (xout,yout,wout)
                yield out
            except Exception as e:
                print("TrainDataGenerator: an exception was raised in batch",b," out of ", self.getNBatches(),', expection: ', e)
                raise e
            
    