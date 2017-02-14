import numpy
import ROOT
from root_numpy import tree2array
# below DeepJet modules
from preprocessing import produceWeigths, meanNormProd, MakeBox, MeanNormApply

import sys
import os

"""
This is an example calling all functiond of preprocessing
"""

inputDataDir = sys.argv[1]
if inputDataDir[-1] != "/":
    inputDataDir+="/"
inputDataName =  sys.argv[2]
inputMeansStd  =  sys.argv[3]
outputFilesTag = sys.argv[4]
outputDir = inputDataDir+outputFilesTag
os.mkdir(outputDir)

# The roofile from DeepNtupler
rfile = ROOT.TFile(inputDataDir+inputDataName)
tree = rfile.Get("deepntuplizer/tree")
Tuple = tree2array(tree)
# Do not trust that the initial *.root is random! Do not do this if you want a validation sample where you recovert the output to root. 
numpy.random.shuffle(Tuple)


# The below filter jets where you genjet troth has Pt < 0 (i.e. PU jets)
# Should only be a sanity check !!! Do not trust chain before!
truth_check = Tuple['gen_pt']
Njets = truth_check.shape[0]
validTruth = truth_check > 0.
# filter by boolian vector
Tuple = Tuple[validTruth]
if Njets != Tuple.shape[0]:
    print (' Please check, jets without genjets conterparts found! This is bad for PT regression !!')

## No we make a files to get the means and std.
#TupleMeanStd =  meanNormProd(Tuple) 
## Typically one would store that, here we make it on the fly
TupleMeanStd =  numpy.load(inputMeansStd)

# sanity checks, would brake easily if wrong means and std are used (dimension check)
BranchList = Tuple.dtype.names
if BranchList != TupleMeanStd.dtype.names:
    print ('Tuple for subtraction and training should match, please check')
    print (len(BranchList), ' ' , len(BranchList))

# now we calculate weights to have flat PT eta distributions
weight_binXPt = numpy.array([0,5,10,15,20,25,30,35,40,45,50,60,70,80,90,110,120,130,140,150,175 ,200,2000],dtype=float)
weight_binYEta = numpy.array([0,.4,.8,1.2,1.6,2.,2.4,5],dtype=float)
weights = produceWeigths(Tuple,"jet_pt","jet_eta",[weight_binXPt,weight_binYEta],classes=['isB','isC','isUDS','isG'])
# dimension check, weight vector must have tuple length
if weights.shape[0] != Tuple.shape[0]:
    print ('Weigts for subtraction and training should match, please check')
    print  (weights.shape[0],' ', Tuple.shape[0])

# Now we select branches of the dataset (tuple) that we want to put into a convolutional network. All these branches much have eta and PT information! Thus we also call them local variables.
name_ = 'Cpfcan_'
PfBranchList = []
Nname_ = 'Npfcan_'
NPfBranchList = []
# loop files (branch) names and select the PF candidates
#for index , name in enumerate(BranchList):
#        print (name)
#        if name_ in name:
#            if 'n_'+name_ in name:
#                pass
#            else:
#                PfBranchList.append(name)
#        if Nname_ in name:
#            if 'n_'+Nname_ in name:
#                pass
#            else:
#                NPfBranchList.append(name)

## No loop, handpicked
PfBranchList =['Cpfcan_etarel','Cpfcan_phirel','Cpfcan_pt','Cpfcan_isMu','Cpfcan_isEl','Cpfcan_VTX_ass','Cpfcan_puppiw']
NPfBranchList = ['Npfcan_etarel','Npfcan_phirel','Npfcan_pt','Npfcan_HadFrac','Npfcan_isGamma']

# No we define the bins for our convolutional network
binX = numpy.array([-.5,-.3,-.1,.1,.3,.5,7],dtype=float)
binY = numpy.array([-.5,-.3,-.1,.1,.3,.5],dtype=float)
# these are the branch names which define the 2D axis
CPFcands = MakeBox([Tuple[PfBranchList], TupleMeanStd],'Cpfcan_etarel','Cpfcan_phirel',binX,binY,10)
NPFCands = MakeBox([Tuple[NPfBranchList] , TupleMeanStd],'Npfcan_etarel','Npfcan_phirel',binX,binY,10)

# Add cgarged and neutral PF candidates
PFCands = numpy.concatenate((NPFCands,CPFcands),axis=3)

#Get MC truth
truth = Tuple[['Delta_gen_pt_WithNu']]
alternative_truth = Tuple[['gen_pt']]

Flavour_truth =  Tuple[['isB','isC','isUDS','isG']]

# Now we collect the global variables (here only PT
PTjets =  Tuple[['jet_pt','jet_eta','QG_ptD','QG_axis2','QG_mult']]
print('final ',PTjets.dtype)

PTjets =  MeanNormApply(PTjets,TupleMeanStd)

print('final ',PTjets.dtype)
#PTjets=PTjets.view('<f4',type=numpy.ndarray(4))

print('final ',PTjets.shape)

# now we save it, the combined covolutional/dense/regression/multiclassification network needs 5 input files
numpy.save(outputDir+"/weights.npy",weights)
numpy.save(outputDir+"/regres_truth.npy",truth)
numpy.save(outputDir+"/regres_alt_truth.npy",alternative_truth)
numpy.save(outputDir+"/local_X.npy",PFCands)
numpy.save(outputDir+"/global_X.npy",PTjets)
numpy.save(outputDir+"/class_truth.npy",Flavour_truth)
