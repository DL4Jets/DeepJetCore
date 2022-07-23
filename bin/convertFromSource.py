#!/usr/bin/env python3
# encoding: utf-8
'''

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


parser = ArgumentParser('program to convert source files to traindata format')
parser.add_argument("-i", help="input file list (required)", metavar="FILE", default='')
parser.add_argument("-o",  help="set output path (required)", metavar="PATH", default='')
parser.add_argument("-c",  help="set output class (required)", metavar="Class")
parser.add_argument("--gpu", help="enable GPU usage for conversion", action='store_true', default=False)
parser.add_argument("-r",  help="set path to snapshot that got interrupted", metavar="FILE", default='')
parser.add_argument("--testdata", action='store_true', help='convert as test data')
parser.add_argument("-n", default='', help="(optional) number of child processes")
parser.add_argument("--nothreads", action='store_true', help='only spawn one process')
parser.add_argument("--checkFiles", action='store_true', help="enables file checking (requires fileIsValid function of TrainData to be defined)")
parser.add_argument("--noRelativePaths", help="Assume input samples are absolute paths with respect to working directory", default=False, action="store_true")
parser.add_argument("--useweightersfrom", default='', help='(for test data or batching) use weighter objects from a different data collection')


parser.add_argument("--inRange", nargs=2, type=int, help="(for batching) input line numbers")
parser.add_argument("--means", action='store_true', help='(for batching) compute only means')
parser.add_argument("--nforweighter", default='500000', help='set number of samples to be used for weighter object creation')
parser.add_argument("--batch", help='(for batching) provide a batch ID to be used')
parser.add_argument("--noramcopy", action='store_true', help='Do not copy input file to /dev/shm before conversion')
parser.add_argument("-v", action='store_true', help='verbose')
parser.add_argument("-q", action='store_true', help='quiet')

# process options
args=parser.parse_args()

#first GPU
if args.gpu:
    import setGPU
    
from DeepJetCore.DataCollection import DataCollection
from DeepJetCore.conversion.conversion import class_options

infile=args.i
outPath=args.o
if (len(infile)<1 or len(outPath)<1) and not len(args.r):
    parser.print_help()
    exit()
class_name=args.c    
recover=args.r
useweightersfrom=args.useweightersfrom
nchilds=args.n
dofilecheck=args.checkFiles
testdata = args.testdata

if args.gpu:
    if (len(nchilds) and int(nchilds)>1) and (not args.nothreads):
        print("WARNING: enabling gpu for conversion and processing multiple files in parallel could be an issue!")

#fileIsValid

if args.batch:
    raise ValueError('batching not implemented at the moment.')

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
elif not recover:
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
dc.no_copy_on_convert = args.noramcopy
dc.istestdata=testdata
if len(nchilds):
    dc.nprocs=int(nchilds)
if args.batch is not None:
    dc.batch_mode = True

traind=None
if class_name in class_options:
    traind = class_options[class_name]
elif not recover:
    print('available classes:')
    for key, val in class_options.items():
        print(key)
    raise Exception('wrong class selection')

if recover:
    dc.recoverCreateDataFromRootFromSnapshot(recover)        
elif args.means:
    dc.convertListOfRootFiles(
        infile, traind, outPath,
        means_only=True,
        output_name='batch_template.djcdc',
        relpath=relpath,
        checkfiles=dofilecheck
    )
else:
    logging.info('Start conversion')
    dc.convertListOfRootFiles(
        infile, traind, outPath, 
        takeweightersfrom=useweightersfrom,
        output_name=(args.batch if args.batch else 'dataCollection.djcdc'),
        relpath=relpath,
        checkfiles=dofilecheck
    )

if args.inRange is not None:
    os.unlink(infile)
