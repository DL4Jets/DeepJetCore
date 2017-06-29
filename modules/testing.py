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

import imp
try:
    imp.find_module('setGPU')
    print('running on GPU')
    import setGPU
except ImportError:
    found = False
    
from pdb import set_trace

class testDescriptor(object):
    
    def __init__(self):
        self.__sourceroots=[]
        self.__predictroots=[]
        self.metrics=[]
        
    def makePrediction(self, model, testdatacollection, outputDir, 
                       ident='', store_labels = False, monkey_class=''): 
        import numpy as np        
        from root_numpy import array2root
        import os
        monkey_class_obj = None
        if monkey_class:
            module, classname = tuple(monkey_class.split(':'))
            _temp = __import__(module, globals(), locals(), [classname], -1) 
            monkey_class_obj = getattr(_temp, classname)
        
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
            if monkey_class_obj is not None:
                testdatacollection.dataclass = monkey_class_obj()
            td=testdatacollection.dataclass
            
            td.readIn(fullpath)
            truthclasses=td.getUsedTruth()
            formatstring = ['prob_%s%s' % (i, ident) for i in truthclasses]
            regressionclasses=[]
            if hasattr(td, 'regressiontargetclasses'):
                regressionclasses=td.regressiontargetclasses
            
            features=td.x
            labels=td.y
            weights=td.w[0]
            #metric=model.evaluate(features, labels, batch_size=10000)
            prediction = model.predict(features)
            if isinstance(prediction, list):
                formatstring.extend(['reg_%s%s' % (i, ident) for i in regressionclasses])
                if prediction[1].shape[1] > len(regressionclasses):
                    raise ValueError('Regression (2nd prediction output) does not match with the provided targets!')
                all_write = np.concatenate(prediction, axis=1)
                if store_labels:
                    all_write = np.concatenate((all_write, labels[0], labels[1]), axis=1)
                    formatstring.extend(truthclasses)
                    formatstring.append('truePt')
            elif prediction.shape[1] == len(truthclasses):
                all_write = prediction
                if store_labels:
                    all_write = np.concatenate((all_write, labels if not isinstance(labels, list) else labels[0]), axis=1)
                    formatstring.extend(truthclasses)
            else:
                formatstring.extend(['reg_%s%s' % (i, ident) for i in regressionclasses])
                if prediction.shape[1] > 2:
                    raise ValueError('Regression output does not match with the provided targets!')
                all_write = prediction
                if store_labels:
                    all_write = np.concatenate((all_write, labels), axis=1)
                    formatstring.append('truePt')

            all_write = np.concatenate([all_write, weights], axis=1)
            formatstring.append('weight')
                
            all_write = np.core.records.fromarrays(np.transpose(all_write), names= ','.join(formatstring))
            array2root(all_write,outputDir+'/'+outrootfilename,"tree",mode="recreate")
            
            #self.metrics.append(metric)
            self.__sourceroots.append(originroot)
            self.__predictroots.append(outputDir+'/'+outrootfilename)
            print(formatstring)
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
        
    
def makeROCs_async(intextfile, name_list, probabilities_list, truths_list, vetos_list,
                    colors_list, outpdffile, cuts='',cmsstyle=False, firstcomment='',secondcomment='',invalidlist=''): 
    
    files=makeASequence(intextfile,len(name_list))
    cuts=makeASequence(cuts,len(name_list))
    probabilities_list=makeASequence(probabilities_list,len(name_list))
    truths_list=makeASequence(truths_list,len(name_list))
    vetos_list=makeASequence(vetos_list,len(name_list))
    invalidlist=makeASequence(invalidlist,len(name_list))
    import c_makeROCs
    
    def worker():
        
        c_makeROCs.makeROCs(files,name_list,
                        probabilities_list,
                        truths_list,
                        vetos_list,
                        colors_list,
                        outpdffile,cuts,cmsstyle, firstcomment,secondcomment,invalidlist)
    
    
    import multiprocessing
    p = multiprocessing.Process(target=worker)
    p.start()
    return p

    # use multiprocessing return thread for waiting option
    
def makePlots_async(intextfile, name_list, variables, cuts, colours,
                     outpdffile, xaxis='',yaxis='',normalized=False,profiles=False,minimum=-1e100,maximum=1e100): 
    
    
    files_list=makeASequence(intextfile,len(name_list))
    variables_list=makeASequence(variables,len(name_list))
    cuts_list=makeASequence(cuts,len(name_list))
    colours_list=makeASequence(colours,len(name_list))
    

    import c_makePlots
    def worker():
        if profiles:
            c_makePlots.makeProfiles(files_list,name_list,
                              variables_list,cuts_list,colours_list,
                                 outpdffile,xaxis,yaxis,normalized,minimum, maximum)
        else:
            c_makePlots.makePlots(files_list,name_list,
                                 variables_list,cuts_list,colours_list,
                                 outpdffile,xaxis,yaxis,normalized,profiles,minimum,maximum)
    
#    return worker()
    import multiprocessing
    p = multiprocessing.Process(target=worker)
    p.start()
    return p     



def make_association(txtfiles, input_branches=None, output_branches=None, limit=None):
    from root_numpy import root2array
    from pandas import DataFrame
    
    #parse associations
    def association(fname):
        return dict(tuple(i.strip().split()) for i in open(fname))
    associations = [association(i) for i in txtfiles]

    #check that the input files are the same
    keys = set(associations[0].keys())
    for i in associations:
        if set(i.keys()) != keys:
            raise ValueError('Association files with different inputs')
    
    #make input lists
    file_lists = [[] for _ in range(len(associations))]
    input_files = []
    for idx, infile in enumerate(associations[0]):
        if limit and idx >= limit: break
        input_files.append(infile)
        for i, association in enumerate(associations):
            file_lists[i].append(association[infile])

    truth = DataFrame(root2array(input_files, branches=input_branches, treename='deepntuplizer/tree'))
    models = [
        DataFrame(root2array(i, branches=output_branches)) for i in file_lists
        ]
    return truth, models
    
    

    
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
