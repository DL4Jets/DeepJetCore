import numpy
import ROOT
from root_numpy import tree2array
# below DeepJet modules
from preprocessing import produceWeigths, meanNormProd, MakeBox, MeanNormApply, MeanNormZeroPad

import sys
import os

"""
    This is an example calling all functiond of preprocessing
    """

inputDataName = sys.argv[1]
inputMeansStd  =  sys.argv[2]
outputDir = sys.argv[3]
os.mkdir(outputDir)

# The roofile from DeepNtupler
rfile = ROOT.TFile(inputDataName)
tree = rfile.Get("deepntuplizer/tree")
Tuple = tree2array(tree)
# If you do not trust that the initial *.root is random! Do not do this if you want a validation sample where you recovert the output to root.
# numpy.random.shuffle(Tuple)

## No we make a files to get the means and std.
#TupleMeanStd =  meanNormProd(Tuple)
## Typically one would store that, i.e. just read it:
TupleMeanStd =  numpy.load(inputMeansStd)

# sanity checks, would brake easily if wrong means and std are used (dimension check)
BranchList = Tuple.dtype.names
if BranchList != TupleMeanStd.dtype.names:
    print ('Tuple for subtraction and training should match, please check')
    print (len(BranchList), ' ' , len(BranchList))
#print (BranchList)

# now we calculate weights to have flat PT eta distributions
# entries per bin (not x-section, i.e. entries/density) will be flattened
weight_binXPt = numpy.array([10,20,25,30,35,40,45,50,60,70,80,90,110,120,130,140,150,175, 2000],dtype=float)
weight_binYEta = numpy.array([0,.4,.8,1.2,1.6,2.,2.4],dtype=float)
weights = produceWeigths(Tuple,"jet_pt","jet_eta",[weight_binXPt,weight_binYEta],classes=['isB','isC','isUDS','isG'])
# dimension check, weight vector must have tuple length
if weights.shape[0] != Tuple.shape[0]:
    print ('Weigts for subtraction and training should match, please check')
    print  (weights.shape[0],' ', Tuple.shape[0])

numpy.save(outputDir+"/weights.npy",weights)

flatBranches = ['jet_pt', 'jet_eta','TagVarCSV_jetNSecondaryVertices', 'TagVarCSV_trackSumJetEtRatio', 'TagVarCSV_trackSumJetDeltaR', 'TagVarCSV_vertexCategory', 'TagVarCSV_trackSip2dValAboveCharm', 'TagVarCSV_trackSip2dSigAboveCharm', 'TagVarCSV_trackSip3dValAboveCharm', 'TagVarCSV_trackSip3dSigAboveCharm', 'TagVarCSV_jetNSelectedTracks', 'TagVarCSV_jetNTracksEtaRel']
tracksBranches = ['TagVarCSVTrk_trackJetDistVal','TagVarCSVTrk_trackPtRel', 'TagVarCSVTrk_trackDeltaR', 'TagVarCSVTrk_trackPtRatio', 'TagVarCSVTrk_trackSip3dSig', 'TagVarCSVTrk_trackSip2dSig', 'TagVarCSVTrk_trackDecayLenVal']
tracksEtaRel = ['TagVarCSV_trackEtaRel']
sv = ['TagVarCSV_vertexMass', 'TagVarCSV_vertexNTracks', 'TagVarCSV_vertexEnergyRatio','TagVarCSV_vertexJetDeltaR','TagVarCSV_flightDistance2dVal', 'TagVarCSV_flightDistance2dSig', 'TagVarCSV_flightDistance3dVal', 'TagVarCSV_flightDistance3dSig']
CMVA_gluon = ['QG_ptD','QG_axis2','QG_mult']

x_global_flat = MeanNormApply(Tuple[flatBranches],TupleMeanStd)
x_tracks = MeanNormZeroPad(Tuple[tracksBranches],TupleMeanStd,6)
x_tracksEtaRel = MeanNormZeroPad(Tuple[tracksEtaRel],TupleMeanStd,4)
x_sv = MeanNormZeroPad(Tuple[sv],TupleMeanStd,1)
x_CMVA_gluon  = MeanNormApply(Tuple[CMVA_gluon],TupleMeanStd)

#print(x_global_flat.shape , x_tracks.shape,' ' , x_tracksEtaRel.shape, ' ', x_sv.shape)
# make to an narray
#x_global_flat = numpy.array(x_global_flat.tolist())
#print (x_global_flat.shape,' ' , x_tracks.shape, x_tracksEtaRel.shape , x_sv.shape)

x_all = numpy.concatenate( (x_global_flat,x_tracks,x_tracksEtaRel,x_sv) , axis=1)
print('This is the shape of the training sample', x_all.shape)
Flavour_truth =  Tuple[['isB','isC','isUDS','isG']]
numpy.save(outputDir+"/global_X.npy",x_all)
numpy.save(outputDir+"/class_truth.npy",Flavour_truth)
numpy.save(outputDir+"/global_gluon_X.npy",numpy.array(x_CMVA_gluon.tolist()))
Pt_truth = Tuple['Delta_gen_pt_WithNu']
numpy.save(outputDir+"/reg_truth.npy",Pt_truth)



