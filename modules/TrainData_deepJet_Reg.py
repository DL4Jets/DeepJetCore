'''
Created on 21 Feb 2017

@author: jkiesele
'''


from TrainData_deepCSV_ST import TrainData_deepCMVA_ST


class TrainData_deepJet_Reg(TrainData_deepCMVA_ST):
    '''
    same as TrainData_deepCSV but with 5 truth labels: UDSG C B leptonicB leptonicB_C
    '''


    def __init__(self):
        '''
        Constructor
        inherits all branches from deepCMVA
        '''
        TrainData_deepCMVA_ST.__init__(self)
        
        self.regtruth='gen_pt'
        self.regreco='jet_pt'
       
    def produceBinWeighter(self,filename):
        from Weighter import Weighter
        import numpy
        weighter=Weighter() 
        Tuple = self.readTreeFromRootToTuple(filename)
        weight_binXPt = numpy.array([10,25,27.5,30,35,40,45,50,60,75,100,125,150,175,200,250,300,
                                     400,500,600,2000],dtype=float)
        weight_binYEta = numpy.array([0,.4,.8,1.2,1.6,2.,2.4],dtype=float)
        
        if self.remove:
            weighter.createRemoveProbabilities(Tuple,"jet_pt","jet_eta",[weight_binXPt,weight_binYEta],
                                               classes=self.truthclasses)
       
        weighter.createBinWeights(Tuple,"jet_pt","jet_eta",[weight_binXPt,weight_binYEta],classes=self.truthclasses)
    
        return weighter
       
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
        
        
