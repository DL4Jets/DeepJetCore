import numpy

"""
author Markus stoye, A collection of tools for data pre-processing in ML for DeepJet. The basic assumption is that Tuple is a recarray where the fiels are the features. 
"""

def produceWeigths(Tuple,nameX,nameY,bins,classes=[],normed=False):
    """
    provides a weight vector to flatten (typically)  PT and eta
    
    Tuple: are the features that need to be reweighted, it needs to be a recarray that had the fields nameX, nameY and classes. The returned weight vector will flatten the Tuple in a 2D plane nameX,nameY for each class (e.g. is B-quark)

    nameX,nameY: names of the fiels that containt the variables to flattedn (usually eta/Pt)

    bins: are the bins used to flatten the distributions. Please use the syntax as in numpy.histogram2d

    classes: These are the one-hot-encoded classes. The weights are made such that each class is flat in 2D

    """
    # stores the weights
    weight = []
    # stores the 2D histograms and there axis
    hists = []
    countMissedJets = 0    
    # if no classes are present just flatten everthing 
    if classes == []:
        hists.append( numpy.histogram2d(Tuple[nameX],Tuple[nameY],bins, normed=True))
    # if classes present, loop ober them and make 2d histogram for each class
    else:
        for label in classes:
            #print 'the labe is ', label
            nameXvec = Tuple[nameX]
            nameYvec = Tuple[nameY]
            valid = Tuple[label] > 0.
            # print  numpy.histogram2d(nameXvec[valid],nameYvec[valid],bins, normed=True) 
            # lease check out numpy.histogram2d for more info
         #   hists += numpy.histogram2d(nameXvec[valid],nameYvec[valid],bins, normed=True)
            w_,_,_ =  numpy.histogram2d(nameXvec[valid],nameYvec[valid],bins, normed=normed)
            hists.append( w_ )
            
    # collect only the fileds we actually need
    Axixandlabel = [nameX, nameY]+ classes
    axisX = bins[0]
    axisY = bins[1]
    
    # loop over jets
    for jet in iter(Tuple[Axixandlabel]):
        # get bins, use first histogram axis
        binX =  getBin(jet[nameX], axisX)
        binY =  getBin(jet[nameY], axisY)
        if classes == []:
            weight.append(1./hists[0][binX][binY])
        else:
            # count if a class was true (should be in one-hot-encoding, but better not trust anyone!
            didappend =0 
            for index, classs in enumerate(classes):
                if 1 == jet[classs]:
                    weight.append(1./hists[index][binX][binY])
                    didappend=1
            if  didappend == 0:
                #print ' WARNING, event found that had no TRUE label '
                # should not happen, but rather kill jet (weight=0) than everything
                # less verbose
                countMissedJets+=1
                weight.append(0)
    if countMissedJets>0:
        print ('WARNING from weight calculator: ', countMissedJets,'/', len(weight), ' had no valid label and got weight 0 (i.e. are ignore, but eat up space and time')
    weight =  numpy.asarray(weight)
    # to get on average weight one
    weight = weight / weight.mean()
    return weight


def meanNormProd(Tuple):
    """
    This function makes a reacarray with the same fields (branches in root talk) as the input tree. The recarray has only two entries, the mean [0] and the st. dev. [1] for each field. If an field was an array the mean of all entries is taken. This is to be used to mean normalize the the input features to ML for faster convergence.
    comment: done seldomly, no need for a quicker solution.
    TO DO: we need to add a parser to JSON to interface to C++ or store straight a C++ map as well.
   
    """
    BranchList = Tuple.dtype.names
    dTypeList = []
    mean = ()
    stddev = ()
    for name in iter (BranchList):
       # check if scalat or array, arrays are stored as object
        if Tuple[[name]].dtype[0]=='object':
 #        if Tuple[name][0].size>1 or 'sv_' in name :
            # makeov it a simple standard array
            chain = numpy.concatenate(Tuple[name])
            # check for crazy entries
            chainsize = chain.size
            chain = chain[numpy.invert( numpy.isinf( chain[:] )) ]
            if chainsize != chain.size:
                print (' There are Inf in the tuple !!! Removed infinities to keep going, but PLEASE CHECK where the fuck they are comming from ')
            mean = mean + (chain.mean(),)
            stddev = stddev+(chain.std(),)
            dTypeList.append((name, float ))
        else:
            #print (mean, Tuple[name][0].size)
            mean =  mean +  (Tuple[name][:].mean(),)
            stddev = stddev+(Tuple[name][:].std(),)
            dTypeList.append((name, float ))
    x = numpy.array([mean, stddev], dtype=dTypeList )
    return x



def MakeHexagonBox():
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
    print (' overflow ! ', value , ' out of range ' , bins)
    return bins.size-2

def MakeSparseBax(Tuples,nameX,nameY,binX,binY,nMaxObj):
    """
    To be coded
    """
    pass

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
    cutCounter = 0
    # basically a loop over all jets
    for jet in iter(Tuple):

#        print jet["Cpfcan_etarel"]
        array = numpy.zeros( (binX.size-1,binY.size-1,nMax) ,dtype=float)

        for index in range ( jet[nameX].size ):
            binx = getBin(jet[nameX][index],binX)
            biny = getBin(jet[nameY][index],binY)
            if not array[binx][biny][0]*(nInput+1) >= nMax:
                for PFindex , varname in enumerate(BranchList):
                    if(varname==nameX):
                    	# this removes the bin boundary
                        array[binx][biny][int(array[binx][biny][0]*nInput)+PFindex+1] = jet[nameX][index]-binX[binx]
               	        # this removes the bin boundary
                    elif(varname==nameY):
                        array[binx][biny][int(array[binx][biny][0]*nInput)+PFindex+1] = jet[nameY][index]-binY[binx]
                    else:
                        stdDev = TuplesMeanDev[varname][1]
                        if stdDev < 0.00001:
                             #print TO DO: PLEASE FIX THIS UPSTREAM, Units of cm^2 are too big for covariance!!!
                             stdDev = 0.00001
                        array[binx][biny][int(array[binx][biny][0]*nInput)+PFindex+1] = ( jet[varname][index] - TuplesMeanDev[varname][0] ) / stdDev  
                   
                # fisrt in array is the number of objects
                array[binx][biny][0] += 1
            else:
                    cutCounter +=1
        BoxList.append(array)
    print(cutCounter, ' times vector was longer than maximum of: ',  nMaxObj)
    return numpy.asarray(BoxList)


def MeanNormApply(Tuple,MeanNormTuple,keepZeros=False):
    """
    The function subtracts the mean and divedes by the std. deviation.
    It is not intended for fields that are arrays. Flexiable array length features are delt with in makeBox. They are automatically zerpatched and mean subtracted
    """
    for field in iter(Tuple.dtype.names):
        if Tuple[field].dtype=='O':
            print ('WARNING: This is means subtraction is not for vectors! The filed is and array. Use MeanNormZeroPad!' )
        Tuple[field] = numpy.subtract(Tuple[field],MeanNormTuple[field][0])
        if keepZeros:
            print ('Need to put in code to add mean back if 0 should be conserved. Actually I am not sure there is a usecase as we do not zero patch like this currently')
        Tuple[field] = numpy.divide(Tuple[field],MeanNormTuple[field][1])
    print (Tuple.dtype)
    return Tuple
 

def MeanNormZeroPad(Tuple,MeanNormTuple,nMax):

    """
    The function subtracts the mean and divides by the std. deviation.
    It is intended for fields that are arrays. They are automatically zerpatched and mean subtracted up to nMax.
    """

    BranchList = Tuple.dtype.names
    nInput = len(BranchList)
    # How long to make the array
    nMax = nMax*nInput

  # loop over jets
    ZeroPadded = []
    for jet in iter(Tuple):
        # per jet one array with all information on the zero padded list of variables, i.e. trackinformations
        array = numpy.zeros(nMax , dtype=float32)
        # loop over the non zeros entries, caution all elements in th list need same length, i.e. a list of track informations per travk
        for index in range ( jet[BranchList[0]].size ):
            # Now for each "track" or alike we look over all the filds (branches) we want to zero pad
            for varIdx , varname in enumerate(BranchList):
                jet[varname][index]
                # overwrite the zeros with the entries, as intialized with zero, the non overwritten reman
                array [index] = numpy.subtract(jet[varIdx][index],MeanNormTuple[name][0])
                array [index] = numpy.divide(array [index],MeanNormTuple[name][1])
                ZeroPadded.append(array)
    # return the list as ndarray
    return numpy.asarray(ZeroPadded)



