import numpy
import ROOT
from root_numpy import tree2array
import sys
from preprocessing import meanNormProd

#this reads the file from root
rfile = ROOT.TFile(sys.argv[1]+".root")
tree = rfile.Get("deepntuplizer/tree")
Tuple = tree2array(tree,stop=100000)
# If a converted file is around:
#Tuple = numpy.load(sys.argv[1]+"ntuple_ttbarTuple.npy")
#print (Tuple.dtype.names)

TupleMeanStd =  meanNormProd(Tuple[0:100000])

numpy.save(sys.argv[1]+"MeansStd.npy",TupleMeanStd)
#print ('done, just doing a simple test now')
#check = numpy.load(sys.argv[1]+"MeansStd.npy").view(numpy.recarray)
#test = check
#print(TupleMeanStd.shape, ' original ',TupleMeanStd['jet_pt'][0])
#print( test.shape, ' reloaded ',test['jet_pt'][0])

