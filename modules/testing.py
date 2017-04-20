'''
Created on 21 Mar 2017

@author: jkiesele
'''



#
#  make training etc..
#  create new module:
#    in: model, .dc file for testing, identifier
#    out: root tree with <ident>_probB, ... , and a descriptor of these new files that links to the old files, too
#
#  create a module to read in one or more descriptors (source ntuples all the same)
#  passes the descriptors and selected branches to compare (probB, newproB, ..) or something to a C++ module
#  C++: creates ROC curves with root
#
#
#

from __future__ import print_function

class testDescriptor(object):
    
    def __init__(self):
        self.__sourceroots=[]
        self.__predictroots=[]
        self.metrics=[]
        
    def makePrediction(self,model,testdatacollection,outputDir,ident=''):
        
        import numpy as np
        
        from root_numpy import array2root
        import os
        
        outputDir=os.path.abspath(outputDir)
        
        if len(ident)>0:
            ident='_'+ident
        
        self.__sourceroots=[]
        self.__predictroots=[]
        self.metrics=[]
        
        for i in range(len(testdatacollection.samples)):
            sample=testdatacollection.samples[i]
            originroot=testdatacollection.originRoots[i]
            outrootfilename=os.path.basename(originroot).split('.')[0]+'_predict'+ident+'.root'
            
            fullpath=testdatacollection.getSamplePath(sample)
            td=testdatacollection.dataclass
            td.readIn(fullpath)
            truthclasses=td.getUsedTruth()
            formatstring=''
            for tc in truthclasses:
                formatstring+='prob_'+tc+ident+','
            formatstring=formatstring[0:-1] #remove last comma
            print(formatstring)
            features=td.x
            labels=td.y
            metric=model.evaluate(features, labels, batch_size=10000)
            
            prediction = model.predict(features)
            all_write = np.core.records.fromarrays(prediction.transpose(), names= formatstring)
            array2root(all_write,outputDir+'/'+outrootfilename,mode="recreate")
            
            self.metrics.append(metric)
            self.__sourceroots.append(originroot)
            self.__predictroots.append(outputDir+'/'+outrootfilename)
            
            print('\ncreated predition friend tree '+outputDir+'/'+outrootfilename+ ' for '+originroot)

    def writeToTextFile(self, outfile):
        '''
        Very simple text file output to use when creating chains with friends.
        Format:
          source0.root prediction0.root
          source1.root prediction1.root
          ...
        '''
        listifle=open(outfile,'w')
        for i in range(len(self.__predictroots)):
            listifle.write(self.__sourceroots[i]+' '+self.__predictroots[i]+'\n')
        listifle.close()
    
def makeASequence(arg,length):
    isseq=(not hasattr(arg, "strip") and
            hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__"))
    out=[]
    if isseq:
        return arg
    else:
        for i in range(length):
            out.append(arg)      
    return out      
        
#just a wrapper
def makeROCs(intextfile, name_list, probabilities_list, truths_list, vetos_list, colors_list, 
             outpdffile, cuts='',cmsstyle=False, firstcomment='',secondcomment=''): 
    
    files=makeASequence(intextfile,len(name_list))
    cuts=makeASequence(cuts,len(name_list))
    probabilities_list=makeASequence(probabilities_list,len(name_list))
    truths_list=makeASequence(truths_list,len(name_list))
    vetos_list=makeASequence(vetos_list,len(name_list))
            
    import c_makeROCs
    c_makeROCs.makeROCs(files,name_list,
                        probabilities_list,
                        truths_list,
                        vetos_list,
                        colors_list,
                        outpdffile,cuts,cmsstyle, firstcomment,secondcomment)
    
def makeROCs_async(intextfile, name_list, probabilities_list, truths_list, vetos_list,
                    colors_list, outpdffile, cuts='',cmsstyle=False, firstcomment='',secondcomment=''): 
    
    files=makeASequence(intextfile,len(name_list))
    cuts=makeASequence(cuts,len(name_list))
    probabilities_list=makeASequence(probabilities_list,len(name_list))
    truths_list=makeASequence(truths_list,len(name_list))
    vetos_list=makeASequence(vetos_list,len(name_list))
    
    def worker():
        import c_makeROCs
        c_makeROCs.makeROCs(files,name_list,
                        probabilities_list,
                        truths_list,
                        vetos_list,
                        colors_list,
                        outpdffile,cuts,cmsstyle, firstcomment,secondcomment)
    
    
    import multiprocessing
    p = multiprocessing.Process(target=worker)
    p.start()
    return p

    # use multiprocessing return thread for waiting option
    
     
    
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