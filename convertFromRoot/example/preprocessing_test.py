import numpy
import ROOT
from root_numpy import tree2array
# below DeepJet modules
from preprocessing import produceWeigths, meanNormProd, MakeBox, MeanNormApply


"""
here should be the test to validate the function of preprocessing

"""

# The roofile from DeepNtupler
rfile = ROOT.TFile('output.root')
tree = rfile.Get("tree")
Tuple = tree2array(tree)

# The below filter jets where you genjet troth has Pt < 0 (i.e. PU jets)
# Should only be a sanity check !!! Do not trust chain before!
truth_check = Tuple['gen_pt']
Njets = truth_check.shape[0]
validTruth = truth_check > 0.
# filter by boolian vector
Tuple = Tuple[validTruth]

# now we calculate weights to have flat PT eta distributions
weight_binXPt = numpy.array([0,5,10,15,20,25,30,35,40,45,50,60,70,80,90,110,120,130,140,150,175 ,200,2000],dtype=float)
weight_binYEta = numpy.array([0,.4,.8,1.2,1.6,2.,2.4,5],dtype=float)
weights = produceWeigths(Tuple,"jet_pt","jet_eta",[weight_binXPt,weight_binYEta],classes=['isB','isC','isUDS','isG'])

#Now we make pure b,c,uds and G tuple
True_b =  Tuple['isUDS'] > 0 
True_c =  Tuple['isC'] > 0 
True_g =  Tuple['isG'] > 0 
weightsb = weights[True_b]
TupleB = Tuple[True_b]
myBs = numpy.histogram2d(TupleB['jet_pt'],TupleB['jet_eta'],bins=[weight_binXPt,weight_binYEta],weights=weightsb)
print myBs[0]

import matplotlib
#if no X11 use below
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.hist2d(TupleB['jet_pt'],TupleB['jet_eta'],bins=[weight_binXPt,weight_binYEta],weights=weightsb)
plt.colorbar()
plt.savefig('check.pdf') 
