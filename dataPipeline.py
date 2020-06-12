
from DeepJetCore.compiled.c_trainDataGenerator import trainDataGenerator


class TrainDataGenerator(trainDataGenerator):
    
    def __init__(self):
        trainDataGenerator.__init__(self)
        
    def feedNumpyData(self):
        
        for b in range(self.getNBatches()):
            try:
                data = self.getBatch()
                
                xout = data.transferFeatureListToNumpy()
                wout = data.transferWeightListToNumpy()
                yout = data.transferTruthListToNumpy()
                
                out = (xout,yout)
                if len(wout)>0:
                    out = (xout,yout,wout)
                yield out
            except Exception as e:
                print("TrainDataGenerator: an exception was raised in batch",b,":", e)
                raise e
            
    