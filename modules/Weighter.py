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
        

    def createBinWeights(self,Tuple,nameX,nameY,bins,classes=[],normed=False):
        '''
        Constructor
        '''
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
        self.Axixandlabel = [nameX, nameY]+ classes
        self.axisX= bins[0]
        self.axisY= bins[1]
        self.nameX=nameX
        self.nameY=nameY
        self.bins=bins
        self.classes=classes
        self.normed=normed
        
        
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

        
        