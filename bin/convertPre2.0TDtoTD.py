#!/usr/bin/env python
# encoding: utf-8
'''

@author:     jkiesele

'''

from argparse import ArgumentParser

parser = ArgumentParser('simple program to convert root tuples to traindata format')
parser.add_argument("infile", help="input \"meta\" file")
# process options
args=parser.parse_args()
infile=args.infile

if not ".meta" in infile:
    print('wrong input file '+infile)
    exit()

from DeepJetCore.TrainData_compat import TrainData
td = TrainData()
td.readIn(infile)
x = td.x
y = td.y
w = td.w
outfile = infile[:-5]
print(outfile)

from DeepJetCore.TrainData import TrainData

tdnew = TrainData()

tdnew.x = x
tdnew.y = y
tdnew.w = w
tdnew.writeToFile(outfile+".djctd")
