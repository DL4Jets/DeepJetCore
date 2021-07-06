#!/usr/bin/env python3
# encoding: utf-8
'''

@author:     jkiesele

'''


from argparse import ArgumentParser
parser = ArgumentParser('merge or split files belonging to a dataCollection differently. The output will be written to the current working directory!')
parser.add_argument("infile", help="input \"dc\" file")
parser.add_argument("nelementsperfile", help="number of entries per file (output), for ragged, maximum number of elements")
parser.add_argument("--randomise", help="randomise order, could be helpful if difference samples need to be mixed", action='store_true')
args=parser.parse_args()


from DeepJetCore.DataCollection import DataCollection
from DeepJetCore.dataPipeline import TrainDataGenerator

infile=args.infile
nbatch=int(args.nelementsperfile)
randomise = args.randomise

dc = DataCollection(infile)
dc2 = DataCollection(infile)
samples = dc.samples

dir = dc.dataDir
if len(dir)<1:
    dir='.'
insamples = [dir+'/'+s for s in samples]

gen = TrainDataGenerator()
gen.setBatchSize(nbatch)
gen.setSkipTooLargeBatches(False)
gen.setFileList(insamples)

if randomise:
    gen.shuffleFileList()

nbatches = gen.getNBatches()

newsamples=[]
for i in range(nbatches):
    newname = str(samples[0][:-6]+"_n_"+str(i)+".djctd")
    newsamples.append(newname)
    ntd = gen.getBatch()
    print(newname)
    ntd.writeToFile(newname)
    print('..written')
    
dc2.samples = newsamples
dc2.writeToFile(infile[:-5]+"_n.djcdc")