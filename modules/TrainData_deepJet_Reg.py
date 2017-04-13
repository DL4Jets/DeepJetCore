'''
Created on 21 Feb 2017

@author: jkiesele
'''


from TrainData_deepCMVA import TrainData_deepCMVA


class TrainData_PF_Reg(TrainData_deepCMVA):
    
    #PLEASE FILL AGAIN OR SIMILAR! WAS NOT COMMITTED
    def __init__(self):
        self.nothing=0

class TrainData_deepJet_Reg(TrainData_deepCMVA):
    '''
    same as TrainData_deepCSV but with 5 truth labels: UDSG C B leptonicB leptonicB_C
    '''


    def __init__(self):
        '''
        Constructor
        inherits all branches from deepCMVA
        '''
        TrainData_deepCMVA.__init__(self)
        
        self.regtruth='gen_pt_WithNu'
        self.regreco='jet_corr_pt'
       
    
       
    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        

        weights,x_all,alltruth,notremoves=self.getFlavourClassificationData(filename,TupleMeanStd, weighter)
        
        Tuple = self.readTreeFromRootToTuple(filename)
        pttruth=Tuple[self.regtruth]
        ptreco=Tuple[self.regreco]
        
        if self.remove:
            print('remove')
            pttruth=pttruth[notremoves > 0]
            ptreco=ptreco[notremoves > 0]
       
        ## additions jan - was missing
        self.nsamples=len(pttruth)
        
        # check if really necssary
        pttruth.reshape(pttruth.shape[0],1)
        ptreco.reshape(ptreco.shape[0],1)
        
        self.w=[weights]
        self.x=[x_all,ptreco]
        self.y=[alltruth, pttruth]
        
        
