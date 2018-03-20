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

colormap=['red'
 , 'blue'
 , 'darkgreen'
 , 'purple'
 , 'darkred'
 , 'darkblue'
 , 'green'
 , 'darkpurple'
 , 'gray']

dashedcolormap=['red','red,dashed'
 , 'blue','blue,dashed'
 , 'darkgreen','darkgreen,dashed'
 , 'purple','purple,dashed'
 , 'darkred','darkred,dashed'
 , 'darkblue','darkblue,dashed'
 , 'green','green,dashed'
 , 'darkpurple','darkpurple,dashed'
 , 'gray','gray,dashed']
    
from pdb import set_trace

class testDescriptor(object):
    
    def __init__(self, addnumpyoutput=False):
        self.__sourceroots=[]
        self.__predictroots=[]
        self.metrics=[]
        self.use_only = []
        self.addnumpyoutput=addnumpyoutput
        
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
            
            if hasattr(td, 'customlabels'):
                import copy
                formatstring=copy.deepcopy(td.customlabels)
            
            else:
                truthclasses=td.getUsedTruth()
                formatstring=[]
                if len(truthclasses)>0 and len(truthclasses[0])>0:
                    formatstring = ['prob_%s%s' % (i, ident) for i in truthclasses]
                regressionclasses=[]
                if hasattr(td, 'regressiontargetclasses'):
                    regressionclasses=td.regressiontargetclasses
                #new implementation. Please check with the store_labels option, Mauro
                formatstring.extend(['reg_%s%s' % (i, ident) for i in regressionclasses])


            features=td.x
            labels=td.y
            weights=td.w[0]
            
            
            
            prediction = model.predict(features)
            if self.use_only:
                prediction = [prediction[i] for i in self.use_only]
            if isinstance(prediction, list):
                all_write = np.concatenate(prediction, axis=1)
            else:
                all_write = prediction
            
            all_write = np.concatenate([all_write, weights], axis=1)
            formatstring.append('weight')
            if not all_write.shape[1] == len(formatstring):
                print(formatstring, ' vs ', all_write.shape[1])
                raise ValueError('Prediction output does not match with the provided targets!')
               
            all_write = np.core.records.fromarrays(np.transpose(all_write), names= ','.join(formatstring))
            array2root(all_write,outputDir+'/'+outrootfilename,"tree",mode="recreate")
            
            #self.metrics.append(metric)
            self.__sourceroots.append(originroot)
            self.__predictroots.append(outputDir+'/'+outrootfilename)
            print(formatstring)
            print('\ncreated predition friend tree '+outputDir+'/'+outrootfilename+ ' for '+originroot)
            if self.addnumpyoutput:
                np.save(outputDir+'/'+outrootfilename+'.npy', all_write)
                
            
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
        if len(arg)==length:
            return arg
        for i in range(length/len(arg)):
            out.extend(arg)
    else:
        for i in range(length):
            out.append(arg)      
    return out      
   
def createColours(colors_list,name_list,nnames=None,extralegend=[]):
    extramulti=1
    if extralegend==None:
        extralegend=[]
    if len(extralegend):
        extramulti=len(extralegend)
    if not nnames:
        nnames=len(name_list)
    if 'auto' in colors_list:
        newcolors=[]
        usemap=colormap
        if 'dashed' in colors_list and not len(extralegend):
            usemap=dashedcolormap
        if len(name_list) > len(usemap)*extramulti:
            raise Exception('colors_list=auto: too many entries, color map too small: '+str(len(name_list))+'/'+str(len(usemap)*extramulti))
        stylecounter=0
        colorcounter=0
        for i in range(len(name_list)):     
            if len(extralegend):
                newcolors.append(usemap[colorcounter] + ','+extralegend[stylecounter].split('?')[0])    
            else:
                newcolors.append(usemap[colorcounter])     
            colorcounter=colorcounter+1
            if colorcounter == nnames:
                colorcounter=0
                stylecounter=stylecounter+1
        
        colors_list=newcolors   
    return   colors_list    
    
def makeROCs_async(intextfile, name_list, probabilities_list, truths_list, vetos_list,
                    colors_list, outpdffile, cuts='',cmsstyle=False, firstcomment='',secondcomment='',
                    invalidlist='',
                    extralegend=None,
                    logY=True,
                    individual=False,
                    xaxis="",
                    nbins=200,
                    treename='deepntuplizer/tree'):#['solid?udsg','hatched?c']): 
    
    import copy
    
    namelistcopy= copy.deepcopy(name_list)
    extralegcopy=copy.deepcopy(extralegend)
    if cmsstyle and extralegcopy==None:
        extralegcopy=['solid?udsg','dashed?c']
        
    if extralegcopy==None:
        extralegcopy=[]
        
    nnames=len(namelistcopy)
    nextra=0
    if extralegcopy:
        nextra=len(extralegcopy)
    

    if nextra>1 and  len(namelistcopy[-1].strip(' ')) >0 :
        extranames=['INVISIBLE']*(nnames)*(nextra-1)
        namelistcopy.extend(extranames)
        
    
    colors_list=createColours(colors_list,namelistcopy,nnames,extralegcopy)   
        
    #check if multi-input file   
    files=makeASequence(intextfile,len(namelistcopy))
    
    
    allcuts=makeASequence(cuts,len(namelistcopy))
    probabilities_list=makeASequence(probabilities_list,len(namelistcopy))
    truths_list=makeASequence(truths_list,len(namelistcopy))
    vetos_list=makeASequence(vetos_list,len(namelistcopy))
    invalidlist=makeASequence(invalidlist,len(namelistcopy))
    
    
    
    from DeepJetCore.compiled import c_makeROCs
    
    
    def worker():
        try:
            c_makeROCs.makeROCs(files,namelistcopy,
                        probabilities_list,
                        truths_list,
                        vetos_list,
                        colors_list,
                        outpdffile,allcuts,cmsstyle, firstcomment,secondcomment,invalidlist,extralegcopy,logY,
                        individual,xaxis,nbins,treename)
        
        except Exception as e:
            print('error for these inputs:')
            print(files)
            print(allcuts)
            print(probabilities_list)
            print(truths_list)
            print(vetos_list)
            print(invalidlist)
            raise e
    
    
    import multiprocessing
    p = multiprocessing.Process(target=worker)
    p.start()
    return p

    # use multiprocessing return thread for waiting option
    
def makePlots_async(intextfile, name_list, variables, cuts, colours,
                     outpdffile, xaxis='',yaxis='',
                     normalized=False,profiles=False,
                     minimum=-1e100,maximum=1e100,widthprofile=False,
                     treename="deepntuplizer/tree",
                     nbins=0,xmin=0,xmax=0,
                     ): 
    
    
    files_list=makeASequence(intextfile,len(name_list))
    variables_list=makeASequence(variables,len(name_list))
    cuts_list=makeASequence(cuts,len(name_list))
    
    colours_list=createColours(colours, name_list)
    
    

    from DeepJetCore.compiled import c_makePlots
    def worker():
        if profiles:
            c_makePlots.makeProfiles(files_list,name_list,
                              variables_list,cuts_list,colours_list,
                                 outpdffile,xaxis,yaxis,normalized,minimum, maximum,treename)
        else:
            c_makePlots.makePlots(files_list,name_list,
                                 variables_list,cuts_list,colours_list,
                                 outpdffile,xaxis,yaxis,normalized,profiles,widthprofile,minimum,maximum,
                                 treename,nbins,xmin,xmax)
    
#    return worker()
    import multiprocessing
    p = multiprocessing.Process(target=worker)
    p.start()
    return p   
  
def makeEffPlots_async(intextfile, name_list, variables, cutsnum,cutsden, colours,
                     outpdffile, xaxis='',yaxis='',
                     minimum=1e100,maximum=-1e100,
                     rebinfactor=1, SetLogY = False, Xmin = 100, Xmax = -100. ,
                     treename="deepntuplizer/tree"): 
    
    
    files_list=makeASequence(intextfile,len(name_list))
    variables_list=makeASequence(variables,len(name_list))
    cutsnum_list=makeASequence(cutsnum,len(name_list))
    cutsden_list=makeASequence(cutsden,len(name_list))
    
    colours_list=createColours(colours, name_list)
    
    

    import c_makePlots
    def worker():
        try:
            c_makePlots.makeEffPlots(files_list,name_list,
                                 variables_list,cutsnum_list,cutsden_list,colours_list,
                                 outpdffile,xaxis,yaxis,rebinfactor,SetLogY, Xmin, Xmax,minimum,maximum,treename)
        except Exception as e:
            print('error for these inputs:')
            print(files_list)
            print(name_list)
            print(variables_list)
            print(cutsnum_list)
            print(cutsden_list)
            print(colours_list)
            raise e
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
    
    

def plotLoss(infilename,outfilename,range):
    
    import matplotlib
    
    matplotlib.use('Agg') 
    
    infile=open(infilename,'r')
    trainloss=[]
    valloss=[]
    epochs=[]
    i=0
    automax=0
    automin=100
    for line in infile:
        if len(line)<1: continue
        tl=float(line.split(' ')[0])
        vl=float(line.split(' ')[1])
        trainloss.append(tl)
        valloss.append(vl)
        epochs.append(i)
        i=i+1
        if i==5:
            automax=max(tl,vl)
        automin=min(automin,vl,tl)
        
    
    import matplotlib.pyplot as plt
    f = plt.figure()
    plt.plot(epochs,trainloss,'r',label='train')
    plt.plot(epochs,valloss,'b',label='val')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    if len(range)==2:
        plt.ylim(range)
    elif automax>0:
        plt.ylim([automin*0.9,automax])
    f.savefig(outfilename)
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
