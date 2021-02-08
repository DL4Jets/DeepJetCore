#!/usr/bin/env python3
# encoding: utf-8
'''

@author:     jkiesele

'''

from argparse import ArgumentParser
import os
parser = ArgumentParser('simple program to convert old data set to the new format')
parser.add_argument("infile", help="input \"dc\" file")
parser.add_argument("--in_place", help="replace files in place: warning, no backups are created",default=False, action="store_true")
# process options
args=parser.parse_args()



from DeepJetCore import DataCollection, TrainData
infile=args.infile

dc=DataCollection(infile)
inpath = dc.dataDir

insamples = [dc.getSamplePath(s) for s in dc.samples]

for s in insamples:
    if not args.in_place:
        os.system('cp '+s+' '+s+'.backup')
    td=TrainData()
    td.readFromFile(s)
    td.writeToFile(s)