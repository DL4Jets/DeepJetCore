#!/usr/bin/env python3
# encoding: utf-8

from argparse import ArgumentParser

parser = ArgumentParser('convert a data collection to a single set of numpy arrays. Warning, this can produce a large output')
parser.add_argument('inputDataCollection')
parser.add_argument('outputFilePrefix')
parser.add_argument("--nfiles",  help="select number of files to be converted, default is every file", type=int, default=-1)
args = parser.parse_args()


from DeepJetCore.DataCollection import DataCollection

print('reading data collection')

dc=DataCollection()
dc.readFromFile(args.inputDataCollection)
nfiles = args.nfiles
print('producing feature array')
feat=dc.getAllFeatures(nfiles=nfiles)

print('producing truth array')
truth=dc.getAllLabels(nfiles=nfiles)

print('producing weight array')
weight=dc.getAllWeights(nfiles=nfiles)


from numpy import save

print('saving output')
for i in range(len(feat)):
    save(args.outputFilePrefix+'_features_'+str(i) +'.npy', feat[i])
    
for i in range(len(truth)):
    save(args.outputFilePrefix+'_truth_'+str(i) +'.npy', truth[i])
    
for i in range(len(weight)):
    save(args.outputFilePrefix+'_weights_'+str(i) +'.npy', weight[i])
    
