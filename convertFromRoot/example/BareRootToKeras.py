import numpy
import ROOT
from root_numpy import tree2array
from  numpy.lib import recfunctions
from itertools import chain
import itertools


def produceWeigths(Tuple,nameX,nameY,bins=100,classes=[]):
    """
    provides a weight vector to flatten PT and eta
    """
    weight = []
    hists = ()
    if classes == []:
        hists = hists + numpy.histogram2d(Tuple[nameX],Tuple[nameY],bins, normed=True)
    else:
        # to be coded, make a list of 
        pass
    #print len(hists)
    Axixandlabel = [nameX, nameY]+ classes
    for jet in iter(Tuple[Axixandlabel]):
        if classes == []:
            binX =  getBin(jet[nameX], hists[1])
            binY =  getBin(jet[nameY], hists[2])
     #       print hists[0].shape, ' ' ,  binX, ' ' ,binY
            weight.append(1./hists[0][binX][binY])
    
    return numpy.asarray(weight)


def meanNormProd(Tuple):
    """
    This function makes a reacarray with the same fields (branches in root talk) as the input tree. The recarray has only two entries, the mean [0] and the st. dev. [1] for each field. If an field was an array the mean of all entries is taken. This is to be usedt  to mean normalize the the input features to ML for faster convergence.
    comment: done seldomly, no need for a quicker solution.
    FIXME: we need to add a parser to JSON to interface to C++ or store straight a C++ map as well.

    """
    BranchList = Tuple.dtype.names
    dTypeList = []
    mean = ()
    stddev = ()
    for name in iter (BranchList):
        print name
       # check if scalat or array
        if(Tuple[name][0].size>1):
            # make it a simple standard array
            chain = numpy.concatenate(Tuple[name])
            # check for crazy entries
            chainsize = chain.size
            chain = chain[numpy.invert( numpy.isinf( chain[:] )) ]
            if chainsize != chain.size:
                print ' There are Inf in the tuple !!! Removed infinities to keep going, but PLEAS CHECK where the fuck they are comming from '
            mean = mean + (chain.mean(),)
            stddev = stddev+(chain.std(),)
            dTypeList.append((name, float ))
        else:
           # print mean
            mean =  mean +  (Tuple[name][:].mean(),)
            stddev = stddev+(Tuple[name][:].std(),)
            dTypeList.append((name, float ))
    x = numpy.array([mean, stddev], dtype=dTypeList )
    return x

def MakeHexagonBins():
    """
    This shoudl return a 1D array with hexagonal coordinates. Zero is the center.
    """

def getBin(value, bins):
    """
    Get the bin of "values" in axis "bins".
    Not forgetting that we have more bin-boundaries than bins (+1) :)
    """
    for index, bin in enumerate (bins):
        # assumes bins in increasing order
        if value < bin:
            return index-1            
    print ' overflow !' 
    return bins.size-2


def MakeBox(Tuples,nameX,nameY,binX,binY,nMaxObj):

    """ 
    function to build from a set of continuously located variables a binned version, e.g. all elements within phi and eta are gathered into one bin. The methods return a 3D ndarray with to direction (e.g. eta, phi) and as 3D (channels) a zero padded constant length vector with the variables
    
    Tuples: list of tupels (recarray) containg only the set of variables that need to be zero padded. First Tuple is the one to be zero padded, that latter should store for mean substraction and scaling as first element. The alttaer can be made with meanNormProd(Tuple)

    nameX, nameY: String with the names on which to apply the cuts
    binX,binY ndarray with the bin boudaries
    nMaxObj: Number of objects per bin. If not reached it will be zero padded. If too long cropped.

    FIXME: many zeros will blow up memory use, should be optimized!!
    """
    Tuple = Tuples[0]
    TuplesMeanDev =  Tuples[1]
    BranchList = Tuple.dtype.names
    nInput = len(BranchList)
    BoxList = []
    # How long to make the array
    nMax = nMaxObj*nInput+1
    
    # basically a loop over all jets
    for jet in iter(Tuple):

#        print jet["Cpfcan_etarel"]
        array = numpy.zeros( (binX.size-1,binY.size-1,nMax) ,dtype=float)

        for index in range ( jet[nameX].size ):
            binx = getBin(jet[nameX][index],binX)
            biny = getBin(jet[nameY][index],binY)
            if not array[binx][biny][0]*(nInput+1) >= nMax:
		for PFindex , varname in enumerate(BranchList):
                     #if numpy.isinf(array[binx][biny][int(array[binx][biny][0]*nInput)+PFindex+1]) : print 'inf'	 	     
	             #if numpy.isnan( array[binx][biny][int(array[binx][biny][0]*nInput)+PFindex+1] ) : print 'nan'
	             if(varname==nameX):
			# this removes the bin boundary
        	        array[binx][biny][int(array[binx][biny][0]*nInput)+PFindex+1] = jet[nameX][index]-binX[binx]
               	        # this removes the bin boundary
		     elif(varname==nameY):
	                array[binx][biny][int(array[binx][biny][0]*nInput)+PFindex+1] = jet[nameY][index]-binY[binx]
		     else:
                         stdDev =  TuplesMeanDev[varname][1]
                         if stdDev < 0.00001:
                             #print  'stdDev to small, PLEASE FIX THIS UPSTREAM'
                             stdDev = 0.00001
                         array[binx][biny][int(array[binx][biny][0]*nInput)+PFindex+1] = ( jet[varname][index] - TuplesMeanDev[varname][0] ) / stdDev  
                         #print varname , ' ',TuplesMeanDev[varname][0] ," ",   TuplesMeanDev[varname][1]
                # fisrt in array is the number of objects
                array[binx][biny][0] += 1
        
  #  print binX.shape, ' ',binX.size
        BoxList.append(array)
    return numpy.asarray(BoxList)


# here a full blown example
rfile = ROOT.TFile("output.root")
tree = rfile.Get("tree")
Tuple = tree2array(tree) 
TupleMeanStd =  meanNormProd(Tuple) 
BranchList = Tuple.dtype.names
# sanity checks
if BranchList != TupleMeanStd.dtype.names:
    print 'Tuple for subtraction and training should match, please check'
    print len(BranchList), ' ' , len(BranchList)
weights = produceWeigths(Tuple,"jet_pt","jet_eta")
if weights.shape[0] != Tuple.shape[0]:
    print 'Weigts for subtraction and training should match, please check'
    print  weights.shape[0],' ', Tuple.shape[0]

name_ = 'Cpfcan'
PfBranchList = []

Nname_ = 'Npfcan'
NPfBranchList = []

for index , name in enumerate(BranchList):
	#print 'branch index: ' ,index, ' , branch name: ',name
        #print 'the type is ' ,Tuple[0][name]
	if name_ in name:
           # print name_ , ' ', name
            if 'n_'+name_ in name:
                pass
         #       print 'counter ', name
            else:
	        PfBranchList.append(name)
        if Nname_ in name:
           # print name_ , ' ', name
            if 'n_'+Nname_ in name:
                print 'counter ', name
            else:
	        NPfBranchList.append(name)
#
#BranchList =PFTuple.dtype.names
#for index , name in enumerate(BranchList):
#      print 'branch index: ' ,index, ' , branch name: ',name
#      print 'the type is ' ,Tuple[name].dtype

binX = numpy.array([-.5,-0.25,0,2.5,.5],dtype=float)
binY = numpy.array([-.5,-0.25,0,25,.5],dtype=float)
nameX = 'Cpfcan_etarel'
nameY = 'Cpfcan_phirel'

CPFcands = MakeBox([Tuple[PfBranchList], TupleMeanStd],nameX,nameY,binX,binY,20)
NPFCands = MakeBox([Tuple[NPfBranchList] , TupleMeanStd],'Npfcan_etarel','Npfcan_phirel',binX,binY,10)

# Add cgarged and neutral PF candidates
PFCands = numpy.concatenate((NPFCands,CPFcands),axis=3)

#Get MC truth
truth = Tuple['gen_pt']
# The below filter events where you genjet troth has Pt < 0 (i.e. PU events)
validTruth = truth > 0.
# filter by boolian vector
truth = truth[validTruth]
PFCands = PFCands[validTruth,:,:,:]
weights = weights[validTruth]

print truth.shape
print PFCands.shape

#import h5py
#h5truth = h5py.File('truth.h5','w')
#h5truth.create_dataset('truth',data=truth)
#h5jets = h5py.File('PFjets.h5','w')
#h5jets.create_dataset('PFjets',data=PFCands)

numpy.save("weights.npy",weights)
numpy.save("truth.npy",truth)
numpy.save("PFjets.npy",PFCands)

#stuff = MakeBox(Tuple)
#print stuff.shape
