import numpy
import ROOT
from root_numpy import tree2array
import sys
from preprocessing import meanNormProd


rfile = ROOT.TFile(sys.argv[1]+".root")
tree = rfile.Get("tree")
Tuple = tree2array(tree)
truth_check = Tuple['gen_pt']
Njets = truth_check.shape[0]
validTruth = truth_check > 0.
## No we make a files to get the means and std.
TupleMeanStd =  meanNormProd(Tuple) 
numpy.save(sys.argv[1]+"MeansStd.npy",TupleMeanStd)

