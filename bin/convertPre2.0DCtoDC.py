#!/usr/bin/env python3
# encoding: utf-8
'''

@author:     jkiesele

'''

from argparse import ArgumentParser

parser = ArgumentParser('simple program to convert old datacollection format to the new one')
parser.add_argument("infile", help="input \"dc\" file")
parser.add_argument("-c",  choices = class_options.keys(), help="set new output class (options: %s)" % ', '.join(class_options.keys()), metavar="Class")

# process options
args=parser.parse_args()


import os
from multiprocessing import Pool
from DeepJetCore.TrainData_compat import TrainData as TDOld
from DeepJetCore.TrainData import TrainData

from DeepJetCore.DataCollection_compat import DataCollection as DCOld
from DeepJetCore.DataCollection import DataCollection
from DeepJetCore.conversion.conversion import class_options 

infile=args.infile

class_name = args.c

if class_name in class_options:
    traind = class_options[class_name]
else:
    print('available classes:')
    for key, val in class_options.iteritems():
        print(key)
    raise Exception('wrong class selection')

if not ".dc" in infile:
    raise Exception('wrong input file '+infile)
    
dir = os.path.dirname(infile)

dcold = DCOld()
dcold.readRawFromFile(infile)


dcnew = DataCollection()
dcnew.dataclass = traind()
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
    
    tdnew.tdnew._store(x,y,w)
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


