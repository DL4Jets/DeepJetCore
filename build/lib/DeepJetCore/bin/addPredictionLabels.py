#!/usr/bin/env python
# encoding: utf-8

from argparse import ArgumentParser
from DeepJetCore.DataCollection import DataCollection

parser = ArgumentParser('add custom prediction labels to a dataCollection. Not necessary in the standard workflow')
parser.add_argument('inputDataCollection')
parser.add_argument('--use', help='comma-separated list of prediction labels to be used')
parser.add_argument('outputDataCollection')
args = parser.parse_args()
if not args.use:
    raise Exception('labels to be injected must be specified')

labels= [i for i in args.use.split(',')]
    
print('reading data collection')

dc=DataCollection()
dc.readFromFile(args.inputDataCollection)
print('adding labels:')
print(labels)
dc.defineCustomPredictionLabels(labels)
dc.writeToFile(args.outputDataCollection)
