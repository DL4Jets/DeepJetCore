
from DeepJetCore.compiled.c_trainDataGenerator import trainDataGenerator
import numpy as np

class TrainDataGenerator(trainDataGenerator):
    
    def __init__(self, 
                 pad_rowsplits=False, 
                 extend_truth_list_by=0,
                 dict_output=False):
        
        trainDataGenerator.__init__(self)
        self.extend_truth_list_by = extend_truth_list_by
        self.pad_rowsplits=pad_rowsplits
        self.dict_output = dict_output
        
        
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
                
                if self.extend_truth_list_by > 0:
                    tadd = [np.array([0],dtype='float32') for _ in range(self.extend_truth_list_by)]
                    if self.dict_output:
                        keyadd = ["_truth_extended_"+str(i) for i in range(self.extend_truth_list_by)]
                        yout.update({k:v for k,v in zip(keyadd,tadd)})
                    else:
                        yout += tadd
                
                out = (xout,yout)
                if len(wout)>0:
                    out = (xout,yout,wout)
                yield out
            except Exception as e:
                print("TrainDataGenerator: an exception was raised in batch",b," out of ", self.getNBatches(),', expection: ', e)
                raise e
            
    def feedTorchTensors(self):
        pass