'''
Created on 26 Feb 2017

@author: jkiesele
'''

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
        self.classes=[]
        self.refclassidx=0
        self.undefTruth=[]
    
    def createRemoveProbabilities(self,Tuple,nameX,nameY,bins,classes,referenceclass='isB'):
        import numpy
        
        referenceidx=0
        try:
            referenceidx=classes.index(referenceclass)
        except:
            print('createRemoveProbabilities: reference index not found in class list')
            raise Exception('createRemoveProbabilities: reference index not found in class list')
               
        
        self.axisX= bins[0]
        self.axisY= bins[1]
        self.nameX=nameX
        self.nameY=nameY
        if len(classes) > 0:
            self.Axixandlabel = [nameX, nameY]+ classes
        else:
            self.Axixandlabel = [nameX, nameY]
        self.bins=bins
        self.classes=classes
        self.refclassidx=referenceidx
        
        
        def getScaler(histo,refhisto):
            scaler=0.
            for indexx,binx in enumerate(self.axisX):
                if not indexx:  continue
                for indexy,biny in enumerate(self.axisY):
                    if not indexy:  continue
                    #print(indexx,indexy)
                    refval=refhisto[indexx-1][indexy-1]
                    thisval=histo[indexx-1][indexy-1]
                    ratio=1.
                    if thisval:
                        ratio=float(refval)/float(thisval)
                    if ratio>scaler: scaler=ratio
            return scaler
        
        def getProbHisto(scaledhisto,refhisto,classname):
            out=numpy.copy(refhisto)
            for indexx,binx in enumerate(self.axisX):
                if not indexx:
                    continue
                for indexy,biny in enumerate(self.axisY):
                    if not indexy:
                        continue
                    refval=refhisto[indexx-1][indexy-1]
                    thisval=scaledhisto[indexx-1][indexy-1]
                    prob=0
                    if thisval+refval:
                        prob=float(thisval-refval)/float(thisval+refval) 
                    if classname in self.undefTruth:
                        prob=1
                    out[indexx-1][indexy-1]=prob
            return out
        

        labeltuple=Tuple[classes]
        ytuple=Tuple[nameY]
        xtuple=Tuple[nameX]
        
        selidxs=[]
        for c in classes:
            selidxs.append(labeltuple[c]>0)
        
    
        refx=xtuple[selidxs[referenceidx]]
        refy=ytuple[selidxs[referenceidx]]
       
        refhist,_,_=numpy.histogram2d(refx,refy,bins)
       
        probhists=[]
        
    
        for i in range(len(classes)):
            tmphist,_,_=numpy.histogram2d(xtuple[selidxs[i]],ytuple[selidxs[i]],bins)
            scaler=getScaler(tmphist,refhist)
            tmphist*=scaler
            probhist=getProbHisto(tmphist,refhist,classes[i])
            probhists.append(probhist)
        
        self.removeProbabilties=probhists
        
        
        
        
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
            
                        
        allxav=0
        validclasses=0            
        for c in range(len(xaverage)):
            if norm[c]:
                allxav+=xaverage[c]/norm[c]
                validclasses+=1
            #print(self.classes[c], c, xaverage[c]/norm[c])
        allxav/=float(validclasses) 
        
        allyav=0  
          
        for c in range(len(yaverage)):
            if norm[c]:
                allyav+=yaverage[c]/norm[c]
        allyav/=float(validclasses)   
             
        for c in range(len(xaverage)):
            if norm[c]:
                reldiff=abs(xaverage[c]/norm[c] - allxav)/allxav
                if reldiff >0.15:
                    print('warning (x) ',self.classes[c],xaverage[c]/norm[c])
                    
         
        for c in range(len(yaverage)):
            if norm[c]:
                reldiff=abs(yaverage[c]/norm[c] - allyav)/allyav
                if reldiff >0.15:
                    print('warning (y) ',self.classes[c],yaverage[c]/norm[c])
                    
            #print(self.classes[c], c, yaverage[c]/norm[c])
            
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

        
        