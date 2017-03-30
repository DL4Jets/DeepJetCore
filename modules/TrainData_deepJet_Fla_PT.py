from TrainData_deepCSV_ST import TrainData_deepCMVA_ST


class TrainData_deepCMVA_Fla_PT(TrainData_deepCMVA_ST):
    '''
    Deep cMVA with 
    '''


    def __init__(self):
        '''
        Constructor
        inherits all branches from deepCMVA
        '''
        TrainData_deepCMVA_ST.__init__(self)
        
        self.regtruth='gen_pt_WithNu'
        self.regCMSSW='jet_corr_pt'

    def readFromRootFile(self,filename,TupleMeanStd, weighter):
        

        weights,x_all,alltruth,notremoves=self.getFlavourClassificationData(filename,TupleMeanStd, weighter)
        
        Tuple = self.readTreeFromRootToTuple(filename)
        pttruth=Tuple[self.regtruth]
        ptrec = Tuple[self.regCMSSW]
        
        if self.remove:
            print('remove')
            pttruth=pttruth[notremoves > 0]
            ptrec=ptrec[notremoves > 0]
       
        
        self.w=[weights,weights]
        self.x=[x_all,ptrec]
        self.y=[alltruth, pttruth]
        
        
