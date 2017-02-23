'''
Created on 21 Feb 2017

@author: jkiesele
'''
from TrainData import TrainData

import numpy
from preprocessing import produceWeigths, MakeBox, MeanNormApply

class TrainData_veryDeepJet(TrainData):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        TrainData.__init__(self)
    
    def readFromRootFile(self,filename,means):
        
        # may want to split this to a more generic function to allow shuffeling later
        # maybe something like "addfromRootFile" -> should go to base class
        
        
        Tuple = self.readTreeFromRootToTuple(filename)
        
        TupleMeanStd =  means
        truth_check = Tuple['gen_pt']
        Njets = truth_check.shape[0]
        validTruth = truth_check > 0.
        # filter by boolian vector
        Tuple = Tuple[validTruth]
        
        if Njets != Tuple.shape[0]:
            print (' Please check, jets without genjets conterparts found! This is bad for PT regression !!')
        
        BranchList = Tuple.dtype.names
        if BranchList != TupleMeanStd.dtype.names:
            print ('Tuple for subtraction and training should match, please check')
            print (len(BranchList), ' ' , len(BranchList))
        
        # now we calculate weights to have flat PT eta distributions
        weight_binXPt = numpy.array([0,5,10,15,20,25,30,35,40,45,50,60,70,80,90,110,120,130,140,150,175 ,200,2000],dtype=float)
        weight_binYEta = numpy.array([0,.4,.8,1.2,1.6,2.,2.4,5],dtype=float)
        
        print ('producing weights')
        
        weights = produceWeigths(Tuple,"jet_pt","jet_eta",[weight_binXPt,weight_binYEta],classes=['isB','isC','isUDS','isG'])
        # dimension check, weight vector must have tuple length
        if weights.shape[0] != Tuple.shape[0]:
            print ('Weigts for subtraction and training should match, please check')
            print  (weights.shape[0],' ', Tuple.shape[0])
        
        self.w=[weights,weights]
        weights=numpy.array([])
        
        PfBranchList =['Cpfcan_etarel','Cpfcan_phirel','Cpfcan_pt','Cpfcan_isMu','Cpfcan_isEl','Cpfcan_VTX_ass','Cpfcan_puppiw']
        NPfBranchList = ['Npfcan_etarel','Npfcan_phirel','Npfcan_pt','Npfcan_HadFrac','Npfcan_isGamma']
        
        # No we define the bins for our convolutional network
        binX = numpy.array([-.5,-.3,-.1,.1,.3,.5,7],dtype=float)
        binY = numpy.array([-.5,-.3,-.1,.1,.3,.5],dtype=float)
        
        print ('making boxes')
        
        # these are the branch names which define the 2D axis
        CPFcands = MakeBox([Tuple[PfBranchList], TupleMeanStd],'Cpfcan_etarel','Cpfcan_phirel',binX,binY,10)
        NPFCands = MakeBox([Tuple[NPfBranchList] , TupleMeanStd],'Npfcan_etarel','Npfcan_phirel',binX,binY,10)
        
        
        print ('concatenate')
        # Add cgarged and neutral PF candidates
        PFCands = numpy.concatenate((NPFCands,CPFcands),axis=3)
        
        #Get MC truth
        truth = Tuple[['Delta_gen_pt_WithNu']]
        # alternative_truth = Tuple[['gen_pt']]
        
        Flavour_truth =  Tuple[['isB','isC','isUDS','isG']]
        
        print ('apply mean to pts')
        # Now we collect the global variables (here only PT
        PTjets =  Tuple[['jet_pt','jet_eta','QG_ptD','QG_axis2','QG_mult']]
        PTjets =  MeanNormApply(PTjets,TupleMeanStd)
        
        #don't copy to not exhaust the memory
        # here, the memory explodes. try to find a smarter solution
        print ('convert to numpy arrays: pfcands')
        #PFCands = numpy.array(PFCands.tolist())
        PFCands = PFCands.view('float32') #numpy.array(PFCands.tolist())
        print ('convert to numpy arrays: ptjets')
        PTjets = numpy.array(PTjets.tolist())
        
        self.x=[PFCands, PTjets]
        PTjets=numpy.array([])
        PFCands =numpy.array([])
        # using view would be quicker but longer syntax
        print ('convert to numpy arrays: flavtruth')
        Flavour_truth = numpy.array(Flavour_truth.tolist())
        print ('convert to numpy arrays: truth')
        truth = numpy.array(truth.tolist())
        self.y=[truth,Flavour_truth]
        
        
        #this is where the actual input to keras is defined. This part should be common for all scenarios
        
        
        
        print('conversion done')
        
        