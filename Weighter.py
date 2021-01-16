'''
Created on 26 Feb 2017
@author: jkiesele
'''

from __future__ import print_function
import numpy as np

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
        self.red_distributions=[]
        self.xedges=[np.array([])]
        self.yedges=[np.array([])]
        self.classes=[]
        self.red_classes=[]
        self.class_weights=[] 
        self.refclassidx=0
        self.undefTruth=[]
        self.truth_red_fusion = []

    def __eq__(self, other):
        'A == B'
        def _all(x):
            if hasattr(x, 'all'):
                return x.all()
            if hasattr(x, '__iter__'):
                return all(x)
            else: return x
            
        def comparator(this, that):
            'compares lists of np arrays'
            return _all((i == j).all() for i,j in zip(this, that))
        
        #empty
        if len(self.Axixandlabel) == len(other.Axixandlabel) and len(self.Axixandlabel) == 0:
            return True
        
        return self.Axixandlabel == other.Axixandlabel and \
           _all(self.axisX == other.axisX) and \
           _all(self.axisY == other.axisY) and \
           comparator(self.hists, other.hists) and \
           comparator(self.removeProbabilties, other.removeProbabilties) and \
           self.classes == other.classes and \
           self.refclassidx == other.refclassidx and \
           self.undefTruth == other.undefTruth and \
           comparator(self.binweights, other.binweights) and \
           comparator(self.distributions, other.distributions) and \
           _all(self.xedges == other.xedges) and \
           _all(self.yedges == other.yedges)
    
    def __ne__(self, other):
        'A != B'
        return not (self == other)
        
    def setBinningAndClasses(self,bins,nameX,nameY,classes, red_classes = -1, truth_red_fusion = -1, method='isB'):

        if method == 'flatten' and red_classes == -1:
            raise Exception('You didnt defined the reduced classes for the flatten method correctly. Create a list with your reduced classes and call it in the setBinningAndClasses function with red_classes = ')
        if method == 'flatten' and truth_red_fusion == -1:
            raise Exception('You didnt defined the fusion for the truth classes for the flatten method correctly. Create a list where each entry is also a list with all the truth classes to fusion into a reduced class. Then call it in the setBinningAndClasses function with thruth_red_fusion = ')

        self.axisX= bins[0]
        self.axisY= bins[1]
        self.nameX=nameX
        self.nameY=nameY
        self.classes=classes
        self.red_classes = red_classes
        self.truth_red_fusion = truth_red_fusion
        if len(self.classes)<1:
            self.classes=['']
        if len(self.red_classes)<1:
            self.red_classes=['']
        if len(self.truth_red_fusion)<1:
            self.truth_red_fusion=['']
        
    def addDistributions(self,Tuple, norm_h = True):
        selidxs=[]
        
        ytuple=Tuple[self.nameY]
        xtuple=Tuple[self.nameX]
        
        useonlyoneclass=len(self.classes)==1 and len(self.classes[0])==0
        
        if not useonlyoneclass:
            labeltuple=Tuple[self.classes]
            for c in self.classes:
                selidxs.append(labeltuple[c]>0)
        else:
            selidxs=[np.zeros(len(xtuple),dtype='int')<1]
                    
        for i, label in enumerate(self.classes):
            #print('axis-X binning :')
            #print(self.axisX)
            #print('axis-Y binning :')
            #print(self.axisY)
            tmphist,xe,ye=np.histogram2d(xtuple[selidxs[i]],ytuple[selidxs[i]],[self.axisX,self.axisY],normed=norm_h)
            self.xedges=xe
            self.yedges=ye
            if len(self.distributions)==len(self.classes):
                self.distributions[i]=self.distributions[i]+tmphist
            else:
                self.distributions.append(tmphist)
                        
    def printHistos(self,outdir):
        def plotHist(hist,outname, histname):
            import matplotlib.pyplot as plt
            H=hist.T
            fig, ax0 = plt.subplots()
            X, Y = np.meshgrid(self.xedges, self.yedges)
            im = ax0.pcolormesh(X, Y, H)
            #fig.colorbar(im, ax=ax)
            if self.axisX[0]>0:
                ax0.set_xscale("log", nonposx='clip')
            else:
                ax0.set_xlim([self.axisX[1],self.axisX[-1]])
                ax0.set_xscale("log", nonposx='mask')
            plt.colorbar(im, ax = ax0)
            ax0.set_title(histname)
            fig.savefig(outname)
            plt.close()
            
        for i in range(len(self.red_classes)):
            if len(self.red_distributions):
                plotHist(self.red_distributions[i],outdir+"/dist_"+self.red_classes[i]+".png",self.red_classes[i]+" distribution")
                #plotHist(self.removeProbabilties[i] ,outdir+"/remprob_"+self.classes[i]+".pdf")
                #plotHist(self.binweights[i],outdir+"/weights_"+self.classes[i]+".pdf")
                #reshaped=self.distributions[i]*self.binweights[i]
                #plotHist(reshaped,outdir+"/reshaped_"+self.classes[i]+".pdf")
        
    def createRemoveProbabilitiesAndWeights(self,referenceclass='isB'):
        
        referenceidx=-1
        
        if referenceclass != 'flatten':
            try:
                referenceidx=self.classes.index(referenceclass)
            except:
                print('createRemoveProbabilities: reference index not found in class list')
                raise Exception('createRemoveProbabilities: reference index not found in class list')
            
        if len(self.classes) > 0 and len(self.classes[0]):
            self.Axixandlabel = [self.nameX, self.nameY]+ self.classes
        else:
            self.Axixandlabel = [self.nameX, self.nameY]
        
        self.refclassidx=referenceidx
        
        refhist=np.zeros((len(self.axisX)-1,len(self.axisY)-1), dtype='float32')
        refhist += 1
        
        if referenceidx >= 0:
            refhist=self.distributions[referenceidx]
            refhist=refhist/np.amax(refhist)
        
        if referenceclass == 'flatten':
            temp = []
            for k in range(len(self.red_classes)):
                temp.append(0)
                for i, label in enumerate(self.classes):
                    if label in self.truth_red_fusion[k]:
                        temp[k] = temp[k] + self.distributions[i]

            for j in range(len(temp)):
                threshold_ = np.median(temp[j][temp[j] > 0]) * 0.01
                nonzero_vals = temp[j][temp[j] > threshold_]
                ref_val = np.percentile(nonzero_vals, 25)

            self.red_distributions = temp
    
        def divideHistos(a,b):
            out=np.array(a)
            for i in range(a.shape[0]):
                for j in range(a.shape[1]):
                    if b[i][j]:
                        out[i][j]=a[i][j]/b[i][j]
                    else:
                        out[i][j]=-10
            return out
                
        reweight_threshold = 15
        max_weight = 1
        raw_hists = {}
        class_events = {}
        result = {}

        probhists=[]
        weighthists=[]

        if referenceclass=='flatten':
            for i, label in enumerate(self.red_classes):
                raw_hists[label] = self.red_distributions[i].astype('float32')
                result[label] = self.red_distributions[i].astype('float32')    
            
            for label, classwgt in zip(self.red_classes, self.class_weights):
                hist = result[label]
                threshold_ = np.median(hist[hist > 0]) * 0.01
                nonzero_vals = hist[hist > threshold_]
                ref_val = np.percentile(nonzero_vals, reweight_threshold)
                # wgt: bins w/ 0 elements will get a weight of 0; bins w/ content<ref_val will get 1
                wgt = np.clip(np.nan_to_num(ref_val / hist, posinf=0), 0, 1)
                result[label] = wgt
                # divide by classwgt here will effective increase the weight later
                class_events[label] = np.sum(raw_hists[label] * wgt) / classwgt
                
            min_nevt = min(class_events.values()) * max_weight
            for label in self.red_classes:
                class_wgt = float(min_nevt) / class_events[label]
                result[label] *= class_wgt
                
            for label in self.classes:
                for i, red_label in enumerate(self.red_classes):
                    if label in self.truth_red_fusion[i]:
                        weighthists.append(result[red_label])
                        probhists.append(1 - result[red_label])                        
                    
            self.removeProbabilties=probhists
            self.binweights=weighthists
        
        else:
            for i in range(len(self.classes)):
                #print(self.classes[i])
                tmphist=self.distributions[i]
                #print(tmphist)
                #print(refhist)
                if np.amax(tmphist):
                    tmphist=tmphist/np.amax(tmphist)
                else:
                    print('Warning: class '+self.classes[i]+' empty.')
                ratio=divideHistos(refhist,tmphist)
                ratio=ratio/np.amax(ratio)#norm to 1
                #print(ratio)
                ratio[ratio<0]=1
                ratio[ratio==np.nan]=1
                ratio = ratio
                weighthists.append(ratio)
                ratio=1-ratio#make it a remove probability
                probhists.append(ratio)

            self.removeProbabilties=probhists
            self.binweights=weighthists

            #make it an average 1
            for i in range(len(self.binweights)):
                self.binweights[i]=self.binweights[i]/np.average(self.binweights[i])
              
    def createNotRemoveIndices(self,Tuple):
        
        if len(self.removeProbabilties) <1:
            raise Exception('removeProbabilties bins not initialised. Cannot create indices per jet')
        
        tuplelength=len(Tuple)
        
        notremove=np.zeros(tuplelength)
        counter=0
        xaverage=[]
        norm=[]
        yaverage=[]
        
        useonlyoneclass=len(self.classes)==1 and len(self.classes[0])==0
        
        for c in self.classes:
            xaverage.append(0)
            norm.append(0)
            yaverage.append(0)
        
        for jet in iter(Tuple[self.Axixandlabel]):
            binX =  self.getBin(jet[self.nameX], self.axisX)
            binY =  self.getBin(jet[self.nameY], self.axisY)
            
            for index, classs in enumerate(self.classes):
                if  useonlyoneclass or 1 == jet[classs]:
                    rand=np.random.ranf()
                    prob = self.removeProbabilties[index][binX][binY]
                    
                    if rand < prob and index != self.refclassidx:
                        #print('rm  ',index,self.refclassidx,jet[classs],classs)
                        notremove[counter]=0
                    else:
                        #print('keep',index,self.refclassidx,jet[classs],classs)
                        notremove[counter]=1
                        xaverage[index]+=jet[self.nameX]
                        yaverage[index]+=jet[self.nameY]
                        norm[index]+=1
            
                    counter += 1
                    break
            else:
                counter += 1
        
            
        if not len(notremove) == counter:
            raise Exception("tuple length must match remove indices length. Probably a problem with the definition of truth classes in the ntuple and the TrainData class")
        
        
        return notremove

    
        
    def getJetWeights(self,Tuple):
        countMissedJets = 0  
        if len(self.binweights) <1:
            raise Exception('weight bins not initialised. Cannot create weights per jet')
        
        weight = np.zeros(len(Tuple))
        jetcount=0
        
        useonlyoneclass=len(self.classes)==1 and len(self.classes[0])==0
        
        for jet in iter(Tuple[self.Axixandlabel]):

            binX =  self.getBin(jet[self.nameX], self.axisX)
            binY =  self.getBin(jet[self.nameY], self.axisY)
            
            for index, classs in enumerate(self.classes):
                if 1 == jet[classs] or useonlyoneclass:
                    weight[jetcount]=(self.binweights[index][binX][binY])
                    
            jetcount=jetcount+1        

        print ('weight average: ',weight.mean())
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
        #print (' overflow ! ', value , ' out of range ' , bins)
        return bins.size-2
