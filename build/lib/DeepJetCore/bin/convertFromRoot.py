#!/usr/bin/env python
# encoding: utf-8
'''
convertFromRoot -- converts the root files produced with the deepJet ntupler to the data format used by keras for the DNN training

convertFromRoot is a small program that converts the root files produced with the deepJet ntupler to the data format used by keras for the DNN training


@author:     jkiesele

'''

import sys
import os

from argparse import ArgumentParser
from pdb import set_trace
import logging
logging.getLogger().setLevel(logging.INFO)

from DeepJetCore.DataCollection import DataCollection

import imp
try:
    imp.find_module('datastructures')
    from datastructures import *
except ImportError:
    print('datastructure modules not found. Please define a DeepJetCore submodule')
   

class_options=[]
import inspect, sys
for name, obj in inspect.getmembers(sys.modules['datastructures']):
    if inspect.isclass(obj) and 'TrainData' in name:
        class_options.append(obj)
      
class_options = dict((str(i).split("'")[1].split('.')[-1], i) for i in class_options)


parser = ArgumentParser('program to convert root tuples to traindata format')
parser.add_argument("-i", help="set input sample description (output from the check.py script)", metavar="FILE")
parser.add_argument("--noRelativePaths", help="Assume input samples are absolute paths with respect to working directory", default=False, action="store_true")
parser.add_argument("-o",  help="set output path", metavar="PATH")
parser.add_argument("-c",  choices = class_options.keys(), help="set output class (options: %s)" % ', '.join(class_options.keys()), metavar="Class")
parser.add_argument("--classArgs",  help="Arguments to pass to output class")
parser.add_argument("-r",  help="set path to snapshot that got interrupted", metavar="FILE", default='')
parser.add_argument("-n", default='', help="(optional) number of child processes")
parser.add_argument("--testdatafor", default='')
parser.add_argument("--usemeansfrom", default='')
parser.add_argument("--nothreads", action='store_true')
parser.add_argument("--means", action='store_true', help='compute only means')
parser.add_argument("--batch", help='Provide a batch ID to be used')
parser.add_argument("-v", action='store_true', help='verbose')
parser.add_argument("-q", action='store_true', help='quiet')

# process options
args=parser.parse_args()
infile=args.i
outPath=args.o
class_name=args.c    
class_args=args.classArgs
recover=args.r
testdatafor=args.testdatafor
usemeansfrom=args.usemeansfrom
nchilds=args.n

if args.batch and not (args.usemeansfrom or args.testdatafor):
    raise ValueError(
        'When running in batch mode you should also '
        'provide a means source through the --usemeansfrom option'
        )

if args.v:
    logging.getLogger().setLevel(logging.DEBUG)
elif args.q:
    logging.getLogger().setLevel(logging.WARNING)

if infile:
    logging.info("infile = %s" % infile)
if outPath:
    logging.info("outPath = %s" % outPath)

# MAIN BODY #
dc = DataCollection(nprocs = (1 if args.nothreads else -1), 
                    useRelativePaths=True if not args.noRelativePaths else False)  
if len(nchilds):
    dc.nprocs=int(nchilds)  

if class_name in class_options:
    traind = class_options[class_name]
elif not recover and not testdatafor:
    print('available classes:')
    for key, val in class_options.iteritems():
        print(key)
    raise Exception('wrong class selection')        
if testdatafor:
    logging.info('converting test data, no weights applied')
    dc.createTestDataForDataCollection(
        testdatafor, infile, outPath, 
        outname = args.batch if args.batch else 'dataCollection.dc',
        batch_mode = bool(args.batch)
    )    
elif recover:
    dc.recoverCreateDataFromRootFromSnapshot(recover)        
elif args.means:
    dc.convertListOfRootFiles(
        infile, traind(class_args) if class_args else traind(), outPath, 
        means_only=True, output_name='batch_template.dc'
        )
else:
    dc.convertListOfRootFiles(
        infile, traind(class_args) if class_args else traind(), outPath, 
        usemeansfrom, output_name = args.batch if args.batch else 'dataCollection.dc',
        batch_mode = bool(args.batch)
        )




