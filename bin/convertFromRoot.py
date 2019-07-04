#!/usr/bin/env python
# encoding: utf-8
'''
convertFromRoot -- converts the root files produced with the deepJet ntupler to the data format used by keras for the DNN training

convertFromRoot is a small program that converts the root files produced with the deepJet ntupler to the data format used by keras for the DNN training


@author:     jkiesele

'''

import sys
import os
import tempfile

from argparse import ArgumentParser
from pdb import set_trace
import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(name)s: %(message)s')
logging.getLogger().setLevel(logging.INFO)

from DeepJetCore.DataCollection import DataCollection
from DeepJetCore.conversion.conversion import class_options

parser = ArgumentParser('program to convert root tuples to traindata format')
parser.add_argument("-i", help="set input sample description (output from the check.py script)", metavar="FILE")
parser.add_argument("--inRange", nargs=2, type=int, help="Input line numbers")
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
parser.add_argument("--nforweighter", default='500000', help='set number of samples to be used for weight and mean calculation')
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

if args.noRelativePaths:
    relpath = ''
else:
    relpath = os.path.dirname(os.path.realpath(infile))

if args.inRange is not None:
    with tempfile.NamedTemporaryFile(delete=False, dir=os.getenv('TMPDIR', '/tmp')) as my_infile:
        with open(infile) as source:
            do_write = False
            for iline, line in enumerate(source):
                if iline == args.inRange[0]:
                    do_write = True
                elif iline == args.inRange[1]:
                    break
                if do_write:
                    path = os.path.realpath(os.path.join(relpath, line))
                    my_infile.write(path)

    infile = my_infile.name
    # new infile will always have absolute path
    relpath = ''

# MAIN BODY #
dc = DataCollection(nprocs = (1 if args.nothreads else -1))
dc.meansnormslimit = int(args.nforweighter)
if len(nchilds):
    dc.nprocs=int(nchilds)
if args.batch is not None:
    dc.batch_mode = True

traind=None
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
        outname=(args.batch if args.batch else 'dataCollection.dc'),
        traind=(traind(class_args) if traind else None),
        relpath=relpath
    )    
elif recover:
    dc.recoverCreateDataFromRootFromSnapshot(recover)        
elif args.means:
    dc.convertListOfRootFiles(
        infile, traind(class_args) if class_args else traind(), outPath,
        means_only=True,
        output_name='batch_template.dc',
        relpath=relpath
    )
else:
    logging.info('Start conversion')
    dc.convertListOfRootFiles(
        infile, traind(class_args) if class_args else traind(), outPath, 
        takemeansfrom=usemeansfrom,
        output_name=(args.batch if args.batch else 'dataCollection.dc'),
        relpath=relpath
    )

if args.inRange is not None:
    os.unlink(infile)
