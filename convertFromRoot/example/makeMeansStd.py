import numpy
import ROOT
from root_numpy import tree2array
import sys
from preprocessing import meanNormProd

print ('a')
rfile = ROOT.TFile(sys.argv[1]+".root")
print ('b')
tree = rfile.Get("deepntuplizer/tree")
print ('c')
Tuple = tree2array(tree)
print ('d')
del tree
print ('e')
numpy.save(sys.argv[1]+"Tuple.npy",Tuple)
#truth_check = Tuple['gen_pt']
#Njets = truth_check.shape[0]
#validTruth = truth_check > 0.
## No we make a files to get the means and std.
#TupleMeanStd =  meanNormProd(Tuple)
#numpy.save(sys.argv[1]+"MeansStd.npy",TupleMeanStd)

