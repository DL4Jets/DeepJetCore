import numpy
#import ROOT
#from root_numpy import tree2array
import sys
from preprocessing import meanNormProd
#rfile = ROOT.TFile(sys.argv[1]+".root")
#tree = rfile.Get("deepntuplizer/tree")
#Tuple = tree2array(tree,stop=100000)
#print ('d')
#del tree
print ('e')
Tuple = numpy.load(sys.argv[1]+"ntuple_ttbarTuple.npy")
print (Tuple.dtype.names)
#truth_check = Tuple['gen_pt']
#Njets = truth_check.shape[0]
#validTruth = truth_check > 0.
## No we make a files to get the means and std.
#TupleMeanStd =  meanNormProd(Tuple[['n_sv', 'sv_pt','jet_pt', 'sv_mass', 'sv_ntracks', 'sv_chi2', 'sv_ndf']])
TupleMeanStd =  meanNormProd(Tuple[0:100000])

print(type(TupleMeanStd))
numpy.save(sys.argv[1]+"MeansStd.npy",TupleMeanStd)
check = numpy.load(sys.argv[1]+"MeansStd.npy").view(numpy.recarray)
test = check #.view(numpy.recarray)
print(TupleMeanStd.shape, ' original ',TupleMeanStd['jet_pt'][0])
print( test.shape, ' reloaded ',test['jet_pt'][0])

