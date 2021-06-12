#!/usr/bin/env python3

from argparse import ArgumentParser


parser = ArgumentParser('Apply a model to a (test) source sample.')
parser.add_argument('inputModel')
parser.add_argument('trainingDataCollection', help="the training data collection. Used to infer data format and batch size.")
parser.add_argument('inputSourceFileList', help="can be text file or a DataCollection file in the same directory as the sample files, or just a single traindata file.")
parser.add_argument('outputDir', help="will be created if it doesn't exist.")
parser.add_argument("-b", help="batch size, overrides the batch size from the training data collection.",default="-1")
parser.add_argument("--gpu",  help="select specific GPU", metavar="OPT", default="")
parser.add_argument("--unbuffered", help="do not read input in memory buffered mode (for lower memory consumption on fast disks)", default=False, action="store_true")
parser.add_argument("--pad_rowsplits", help="pad the row splits if the input is ragged", default=False, action="store_true")


args = parser.parse_args()
batchsize = int(args.b)


import imp
from DeepJetCore.DataCollection import DataCollection
from DeepJetCore.dataPipeline import TrainDataGenerator
import tempfile
import atexit
import os
from keras.models import load_model
from keras import backend as K
from DeepJetCore.customObjects import get_custom_objects
from DeepJetCore.training.gpuTools import DJCSetGPUs

inputdatafiles=[]
inputdir=None

## prepare input lists for different file formats

if args.inputSourceFileList[-6:] == ".djcdc":
    print('reading from data collection',args.inputSourceFileList)
    predsamples = DataCollection(args.inputSourceFileList)
    inputdir = predsamples.dataDir
    for s in predsamples.samples:
        inputdatafiles.append(s)
        
elif args.inputSourceFileList[-6:] == ".djctd":
    inputdir = os.path.abspath(os.path.dirname(args.inputSourceFileList))
    infile = os.path.basename(args.inputSourceFileList)
    inputdatafiles.append(infile)
else:
    print('reading from text file',args.inputSourceFileList)
    inputdir = os.path.abspath(os.path.dirname(args.inputSourceFileList))
    with open(args.inputSourceFileList, "r") as f:
        for s in f:
            inputdatafiles.append(s.replace('\n', '').replace(" ",""))
        

DJCSetGPUs(args.gpu)

custom_objs = get_custom_objects()

model=load_model(args.inputModel, custom_objects=custom_objs)
dc = None
if args.inputSourceFileList[-6:] == ".djcdc" and not args.trainingDataCollection[-6:] == ".djcdc":
    dc = DataCollection(args.inputSourceFileList)
    if batchsize < 1:
        batchsize = 1
    print('No training data collection given. Using batch size of',batchsize)
else:
    dc = DataCollection(args.trainingDataCollection)

outputs = []
os.system('mkdir -p '+args.outputDir)


for inputfile in inputdatafiles:
    
    print('predicting ',inputdir+"/"+inputfile)
    
    use_inputdir = inputdir
    if inputfile[0] == "/":
        use_inputdir=""
    outfilename = "pred_"+os.path.basename( inputfile )
    
    td = dc.dataclass()

    if inputfile[-5:] == 'djctd':
        if args.unbuffered:
            td.readFromFile(use_inputdir+"/"+inputfile)
        else:
            td.readFromFileBuffered(use_inputdir+"/"+inputfile)
    else:
        print('converting '+inputfile)
        td.readFromSourceFile(use_inputdir+"/"+inputfile, dc.weighterobjects, istraining=False)
    

    gen = TrainDataGenerator()
    if batchsize < 1:
        batchsize = dc.getBatchSize()
    print('batch size',batchsize)
    gen.setBatchSize(batchsize)
    gen.setSquaredElementsLimit(dc.batch_uses_sum_of_squares)
    gen.setSkipTooLargeBatches(False)
    gen.setBuffer(td)
    
    predicted = model.predict_generator(gen.feedNumpyData(),
                                        steps=gen.getNBatches(),
                                        max_queue_size=1,
                                        use_multiprocessing=False,verbose=1)
    
    
    x = td.transferFeatureListToNumpy(args.pad_rowsplits)
    w = td.transferWeightListToNumpy(args.pad_rowsplits)
    y = td.transferTruthListToNumpy(args.pad_rowsplits)
    
    td.clear()
    gen.clear()
    
    if not type(predicted) == list: #circumvent that keras return only an array if there is just one list item
        predicted = [predicted]   
    overwrite_outname = td.writeOutPrediction(predicted, x, y, w, args.outputDir + "/" + outfilename, use_inputdir+"/"+inputfile)
    if overwrite_outname is not None:
        outfilename = overwrite_outname
    outputs.append(outfilename)
    
    
    
with open(args.outputDir + "/outfiles.txt","w") as f:
    for l in outputs:
        f.write(l+'\n')
    
    
    
    
    
