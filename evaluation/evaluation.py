'''
Created on 21 Mar 2017

@author: jkiesele
'''


    
    
def plotLoss(infilename,outfilename,range):
    
    import matplotlib
    #matplotlib.use('Agg') 
    import matplotlib.pyplot as plt
    
    infile=open(infilename,'r')
    trainloss=[]
    valloss=[]
    epochs=[]
    i=0
    automax=0
    automin=100
    nlines=0
    with open(infilename,'r') as tmpfile:
        for line in tmpfile:
            if len(line)<1: continue
            nlines+=1
        
    for line in infile:
        if len(line)<1: continue
        tl=float(line.split(' ')[0])
        vl=float(line.split(' ')[1])
        trainloss.append(tl)
        valloss.append(vl)
        epochs.append(i)
        i=i+1
        if i - float(nlines)/2. > 1.:
            automax=max(automax,tl,vl)
        automin=min(automin,vl,tl)
        
    
    
    plt.plot(epochs,trainloss,'r',label='train')
    plt.plot(epochs,valloss,'b',label='val')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    if len(range)==2:
        plt.ylim(range)
    elif automax>0:
        plt.ylim([automin*0.9,automax*1.1])
    #plt.show()
    plt.savefig(outfilename, format='pdf') #why does this crash?
    plt.close()


def plotBatchLoss(infilename,outfilename,range):
    
    import matplotlib
    #matplotlib.use('Agg') 
    import matplotlib.pyplot as plt
    
    infile=open(infilename,'r')
    trainloss=[]
    batch=[]
    i=0
    automax=0
    automin=100
    nlines=0
    with open(infilename,'r') as tmpfile:
        for line in tmpfile:
            if len(line)<1: continue
            nlines+=1
        
    for line in infile:
        if len(line)<1: continue
        tl=float(line.split(' ')[0])
        trainloss.append(tl)
        batch.append(i)
        i=i+1
        if i - float(nlines)/2. > 1.:
            automax=max(automax,tl)
        automin=min(automin,tl)
        
    
    
    plt.plot(batch,trainloss,'r',label='train')
    plt.ylabel('loss')
    plt.xlabel('batch')
    plt.legend()
    plt.ylim([0,6.2])
    #plt.show()
    plt.savefig(outfilename) #why does this crash?
    plt.close()
    
######### old part - keep for reference, might be useful some day 

#just a collection of what will be helpful

#import numpy as np
#from numpy.lib.recfunctions import merge_arrays
#dt1 = [('foo', int), ('bar', float)]
#dt2 = [('foobar', int), ('barfoo', float)]
#aa = np.empty(6, dtype=dt1).view(np.recarray)
#bb = np.empty(6, dtype=dt2).view(np.recarray)
#
#cc = merge_arrays((aa, bb), asrecarray=True, flatten=True)
#type(cc)
#print (cc)
#
#
## this can be used to add new 'branches' to the input array for a root output tuple
##in traindata
#
##only save them for test data (not for val or train)
#passthrough=['jet_pt','jet_eta','pfCombinedInclusiveSecondaryVertexV2BJetTags'] #etc
#savedpassarrays=[]
#for i in range (len(passthrough)):
#    savedpassarrays[i]
#    
#
##here come the truth and predicted parts again, weights for convenience
#all_write = np.core.records.fromarrays(  np.hstack(((predict_test,labels),savedpassarrays)).transpose(), 
#                                             names='probB, probC, probUDSG, isB, isC, isUDSG, [passthrough]')
#                                            #can probably go formats = 'float32,float32,float32,float32,float32,float32,[npt * float32]')
#
#
