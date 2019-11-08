#!/usr/bin/env python
# encoding: utf-8
'''

@author:     jkiesele

'''

import os
from multiprocessing import Pool
from argparse import ArgumentParser
from DeepJetCore.TrainData_compat import TrainData as TDOld
from DeepJetCore.TrainData import TrainData

from DeepJetCore.DataCollection_compat import DataCollection as DCOld
from DeepJetCore.DataCollection import DataCollection 

parser = ArgumentParser('simple program to convert old datacollection format to the new one')
parser.add_argument("infile", help="input \"meta\" file")
# process options
args=parser.parse_args()
infile=args.infile

if not ".dc" in infile:
    print('wrong input file '+infile)
    exit()
    
dir = os.path.dirname(infile)

dcold = DCOld()
dcold.readFromFile(infile)


dcnew = DataCollection()
dcnew.samples = [s[:-4]+'djctd' for s in dcold.samples]
print(dcnew.samples)
dcnew.sourceList = dcold.originRoots
# leave traindata undefined no way to convert.
dcnew.__nsamples = 0 # determine again, also check

outfile = infile[:-2] +'djcdc'
print("infile: ", infile, " outfile", outfile)

def worker(i):

    td = TDOld()
    tdnew = TrainData()
    print("converting",dcold.samples[i])
    
    td.readIn(dir + dcold.samples[i])
    x = td.x
    y = td.y
    w = td.w
    
    tdnew.x = x
    tdnew.y = y
    tdnew.w = w
    tdnew.writeToFile(dcnew.samples[i])
    
    td.clear()
    tdnew.clear()
    del x,y,w
    return True
    
p = Pool()
ret = p.map(worker, range(len(dcold.samples)))

for r in ret:
    if not r:
        print('something went wrong ')
        exit()
    
dcnew.writeToFile(outfile)


