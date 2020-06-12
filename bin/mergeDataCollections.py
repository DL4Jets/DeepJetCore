#!/bin/env python3

from argparse import ArgumentParser
from DeepJetCore.DataCollection import DataCollection
import os

def sumDCandWrite(filelist, outname):
    alldc=[]
    for f in filelist:
        try:
            dc = DataCollection(f)
        except:
            print('read in of '+f +' not working, skip')
            continue
        alldc.append(dc)
        rel  = os.path.relpath(dc.dataDir,os.getcwd())
        dc.prependToSampleFiles(rel+'/')
        dc.dataDir=os.getcwd()

    merged = sum(alldc)
    print(outname)
    merged.writeToFile(outname)

parser = ArgumentParser('program to merge dataCollection files')
parser.add_argument('inputfiles', metavar='N', type=str, nargs='+',help='input data collection files (.dc)')
parser.add_argument("--testsplit", help="The fraction used to create a testing dataset", default=0, type=float)
parser.add_argument("--outputprefix", help="prefix to be used for output", default="merged", type=str)

args = parser.parse_args()

outprefix = args.outputprefix
if len(outprefix) and outprefix[-1] != '_':
    outprefix+='_'

if args.testsplit > 1 or args.testsplit < 0:
    print('testsplit must not be larger than 1 or smaller than 0, abort')
    exit(-1)

#DEBUG
ninput = float(len(args.inputfiles))

if args.testsplit > 0 and ninput*(args.testsplit) < 1:
    print('testsplit too small to produce a single test file, abort')
    exit(-2)
    

trainfiles = []
testfiles  = []

if args.testsplit == 0:
    trainfiles = args.inputfiles
else:
    
    for i in range(len(args.inputfiles)):
        if i < ninput*(1.-args.testsplit):
            trainfiles.append(args.inputfiles[i])
        else:
            testfiles.append(args.inputfiles[i])
            
            
print(trainfiles)
print(testfiles)

if args.testsplit > 0:
    sumDCandWrite(testfiles, outprefix+'test.dc')
sumDCandWrite(trainfiles, outprefix+'train.dc')





