#!/usr/bin/env python3

from argparse import ArgumentParser


parser = ArgumentParser('Apply a model to a (test) source sample and create friend trees to inject it inthe original ntuple')
parser.add_argument('inputModel')
parser.add_argument('trainingDataCollection')
parser.add_argument('inputSourceFileList')
parser.add_argument('outputDir')
parser.add_argument("-b", help="batch size ",default="-1")


args = parser.parse_args()


from DeepJetCore.DataCollection import DataCollection
from DeepJetCore.compiled.c_trainDataGenerator import trainDataGenerator
from DeepJetCore.customObjects import get_custom_objects
from keras.models import load_model
import tempfile
import atexit
import os


batchsize = int(args.b)
 
custom_objs = get_custom_objects()

model=load_model(args.inputModel, custom_objects=custom_objs)
dc = DataCollection(args.trainingDataCollection)
td = dc.dataclass()
outputs = []
inputdir = os.path.abspath(os.path.dirname(args.inputSourceFileList))
os.system('mkdir -p '+args.outputDir)

with open(args.inputSourceFileList, "r") as f:
    for inputfile in f:
        inputfile = inputfile.replace('\n', '')
        use_inputdir = inputdir
        if inputfile[0] == "/":
            use_inputdir=""
        outfilename = "pred_"+os.path.basename( inputfile )
        

        if inputfile[-5:] == 'djctd':
            td.readFromFile(use_inputdir+"/"+inputfile)
        else:
            print('converting '+inputfile)
            td.readFromSourceFile(use_inputdir+"/"+inputfile, dc.weighterobjects, istraining=False)
        
        print('predicting ',inputfile)
        print('batch size',dc.getBatchSize())
        gen = trainDataGenerator()
        gen.setBatchSize(dc.getBatchSize())
        gen.setSquaredElementsLimit(dc.batch_uses_sum_of_squares)
        gen.setSkipTooLargeBatches(False)
        gen.setBuffer(td)
        
        def genfunc():
            while(1):
                d = gen.getBatch()
                yield d.transferFeatureListToNumpy() , d.transferTruthListToNumpy()
                
        predicted = model.predict_generator(genfunc(),
                                            steps=gen.getNBatches(),
                                            max_queue_size=1,
                                            use_multiprocessing=False,verbose=1)
        
        
        x = td.transferFeatureListToNumpy()
        w = td.transferWeightListToNumpy()
        y = td.transferTruthListToNumpy()
        
        td.clear()
        gen.clear()
        
        if not type(predicted) == list: #circumvent that keras return only an array if there is just one list item
            predicted = [predicted]   
        td.writeOutPrediction(predicted, x, y, w, args.outputDir + "/" + outfilename, inputfile)
        
        outputs.append(outfilename)
        
    
    
with open(args.outputDir + "/outfiles.txt","w") as f:
    for l in outputs:
        f.write(l+'\n')
    
    
    
    
    
