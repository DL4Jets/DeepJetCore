'''
Created on 26 Feb 2017

@author: jkiesele
'''

from __future__ import print_function

import matplotlib
#if no X11 use below
matplotlib.use('Agg')

class Weighter(object):
    '''
    contains the histograms/input to calculate jet-wise weights
    '''
    def __init__(self):

        self.Axixandlabel=[]
        self.axisX=[]
        self.axisY=[]
        self.hists =[]
        self.removeProbabilties=[]
        self.binweights=[]
        self.distributions=[]
        self.xedges=[]
        self.yedges=[]
        self.classes=[]
        self.refclassidx=0
        self.undefTruth=[]
    
    def __eq__(self, other):
        'A == B'
        def comparator(this, that):
            'compares lists of np arrays'
            return all((i == j).all() for i,j in zip(this, that))
        
        return self.Axixandlabel == other.Axixandlabel and \
           all(self.axisX == other.axisX) and \
           all(self.axisY == other.axisY) and \
           comparator(self.hists, other.hists) and \
           comparator(self.removeProbabilties, other.removeProbabilties) and \
           self.classes == other.classes and \
           self.refclassidx == other.refclassidx and \
           self.undefTruth == other.undefTruth and \
           comparator(self.binweights, other.binweights) and \
           comparator(self.distributions, other.distributions) and \
           (self.xedges == other.xedges).all() and \
           (self.yedges == other.yedges).all()
    
    def __ne__(self, other):
        'A != B'
        return not (self == other)
        
    def setBinningAndClasses(self,bins,nameX,nameY,classes):
        self.axisX= bins[0]
        self.axisY= bins[1]
        self.nameX=nameX
        self.nameY=nameY
        self.classes=classes
        
    def addDistributions(self,Tuple):
        import numpy
        selidxs=[]
        labeltuple=Tuple[self.classes]
        ytuple=Tuple[self.nameY]
        xtuple=Tuple[self.nameX]
        
        for c in self.classes:
            selidxs.append(labeltuple[c]>0)
            
        for i in range(len(self.classes)):
            tmphist,xe,ye=numpy.histogram2d(xtuple[selidxs[i]],ytuple[selidxs[i]],[self.axisX,self.axisY],normed=True)
            self.xedges=xe
            self.yedges=ye
            if len(self.distributions)==len(self.classes):
                self.distributions[i]=self.distributions[i]+tmphist
            else:
                self.distributions.append(tmphist)
            
    def printHistos(self,outdir):
        import numpy
        def plotHist(hist,outname):
            import matplotlib.pyplot as plt
            H=hist.T
            fig = plt.figure()
            ax = fig.add_subplot(111)
            X, Y = numpy.meshgrid(self.xedges, self.yedges)
            ax.pcolormesh(X, Y, H)
            ax.set_xscale("log", nonposx='clip')
            #plt.colorbar()
            fig.savefig(outname)
            plt.close()
            
        for i in range(len(self.classes)):
            if len(self.distributions):
                plotHist(self.distributions[i],outdir+"/dist_"+self.classes[i]+".pdf")
                plotHist(self.removeProbabilties[i] ,outdir+"/remprob_"+self.classes[i]+".pdf")
                plotHist(self.binweights[i],outdir+"/weights_"+self.classes[i]+".pdf")
                reshaped=self.distributions[i]*self.binweights[i]
                plotHist(reshaped,outdir+"/reshaped_"+self.classes[i]+".pdf")
            
        
    def createRemoveProbabilitiesAndWeights(self,referenceclass='isB'):
        import numpy
        referenceidx=0
        try:
            referenceidx=self.classes.index(referenceclass)
        except:
            print('createRemoveProbabilities: reference index not found in class list')
            raise Exception('createRemoveProbabilities: reference index not found in class list')
        
        if len(self.classes) > 0:
            self.Axixandlabel = [self.nameX, self.nameY]+ self.classes
        else:
            self.Axixandlabel = [self.nameX, self.nameY]
        
        self.refclassidx=referenceidx
        
        
       
        refhist=self.distributions[referenceidx]
        refhist=refhist/numpy.amax(refhist)
        
    
        def divideHistos(a,b):
            out=numpy.array(a)
            for i in range(a.shape[0]):
                for j in range(a.shape[1]):
                    if b[i][j]:
                        out[i][j]=a[i][j]/b[i][j]
                    else:
                        out[i][j]=-10
            return out
                
        probhists=[]
        weighthists=[]
        for i in range(len(self.classes)):
            tmphist=self.distributions[i]
            tmphist=tmphist/numpy.amax(tmphist)
            ratio=divideHistos(refhist,tmphist)
            ratio=ratio/numpy.amax(ratio)#norm to 1
            ratio[ratio<0]=1
            weighthists.append(ratio)
            ratio=1-ratio#make it a remove probability
            probhists.append(ratio)
        
        self.removeProbabilties=probhists
        self.binweights=weighthists
        for h in self.binweights:
            h=h/numpy.average(h)
    
    
        
        
    def createNotRemoveIndices(self,Tuple):
        import numpy
        if len(self.removeProbabilties) <1:
            print('removeProbabilties bins not initialised. Cannot create indices per jet')
            raise Exception('removeProbabilties bins not initialised. Cannot create indices per jet')
        
        tuplelength=len(Tuple)
        
        notremove=[]
        xaverage=[]
        norm=[]
        yaverage=[]
        for c in self.classes:
            xaverage.append(0)
            norm.append(0)
            yaverage.append(0)

        for jet in iter(Tuple[self.Axixandlabel]):
            binX =  self.getBin(jet[self.nameX], self.axisX)
            binY =  self.getBin(jet[self.nameY], self.axisY)
            
            for index, classs in enumerate(self.classes):
                if 1 == jet[classs]:
                    rand=numpy.random.ranf()
                    prob = self.removeProbabilties[index][binX][binY]

                    if rand < prob and index != self.refclassidx:
                        #print('rm  ',index,self.refclassidx,jet[classs],classs)
                        notremove.append(0)
                    else:
                        #print('keep',index,self.refclassidx,jet[classs],classs)
                        notremove.append(1)
                        xaverage[index]+=jet[self.nameX]
                        yaverage[index]+=jet[self.nameY]
                        norm[index]+=1
            
                        
        
            
        if not len(notremove) == tuplelength:
            raise Exception("tuple length must match remove indices length. Probably a problem with the definition of truth classes in the ntuple and the TrainData class")
        
        
        return numpy.array(notremove)

    def createBinWeights(self,Tuple,nameX,nameY,bins,classes=[],normed=False):
        
       
        
        import numpy
        self.Axixandlabel=[]
        self.axisX=[]
        self.axisY=[]
        self.hists =[]
        self.nameX=''
        self.nameY=''
        self.bins=[]
        self.classes=[]
        self.normed=True
        self.hists=[]
        
        if len(classes) > 0:
            self.Axixandlabel = [nameX, nameY]+ classes
        else:
            self.Axixandlabel = [nameX, nameY]
        self.axisX= bins[0]
        self.axisY= bins[1]
        self.nameX=nameX
        self.nameY=nameY
        self.bins=bins
        self.classes=classes
        self.normed=normed
        
    # if no classes are present just flatten everthing 
        if classes == []:
            self.hists.append( numpy.histogram2d(Tuple[nameX],Tuple[nameY],bins, normed=True))
        # if classes present, loop ober them and make 2d histogram for each class
        else:
            for label in classes:
                #print 'the labe is ', label
                nameXvec = Tuple[nameX]
                nameYvec = Tuple[nameY]
                valid = Tuple[label] > 0.
                # print  numpy.histogram2d(nameXvec[valid],nameYvec[valid],bins, normed=True) 
                # lease check out numpy.histogram2d for more info
                # hists += numpy.histogram2d(nameXvec[valid],nameYvec[valid],bins, normed=True)
                w_,_,_ =  numpy.histogram2d(nameXvec[valid],nameYvec[valid],bins, normed=normed)
                self.hists.append( w_ )
                
        # collect only the fileds we actually need
        
        
        
    def getJetWeights(self,Tuple):
        import numpy
        countMissedJets = 0  
        if len(self.hists) <1:
            print('weight bins not initialised. Cannot create weights per jet')
            raise Exception('weight bins not initialised. Cannot create weights per jet')
        
        weight = []
        for jet in iter(Tuple[self.Axixandlabel]):
        # get bins, use first histogram axis
            binX =  self.getBin(jet[self.nameX], self.axisX)
            binY =  self.getBin(jet[self.nameY], self.axisY)
            if self.classes == []:
                weight.append(1./self.hists[0][binX][binY])
            else:
                # count if a class was true (should be in one-hot-encoding, but better not trust anyone!
                didappend =0 
            
                for index, classs in enumerate(self.classes):
                    # print ('ha ',classs , ' ' , 'jet[classs] is ', jet[classs])
                    if 1 == jet[classs]:
                        # print ('is one')
                        weight.append(1./self.hists[index][binX][binY])
                        #if 1./self.hists[index][binX][binY] > 10.*0.0002646:
                        #    print (classs, ' ' , jet[self.nameX], ' ' , jet[self.nameY], ' weight ',  1./self.hists[index][binX][binY]/0.0002646)
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
        print ('weight average: ',weight.mean())
        weight = weight / weight.mean()
        print ('rescaled weight average: ',weight.mean())
        print ('rescaled weight stddev: ',weight.std())
        return weight
        
        
    def getBin(self,value, bins):
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

        
        
