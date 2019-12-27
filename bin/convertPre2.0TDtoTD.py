#!/usr/bin/env python3
# encoding: utf-8
'''

@author:     jkiesele

'''

from argparse import ArgumentParser
from DeepJetCore.conversion.conversion import class_options

parser = ArgumentParser('simple program to convert old (pre 2.0) traindata format to the new one')
parser.add_argument("infile", help="input \"meta\" file")
parser.add_argument("-c",  choices = class_options.keys(), help="set new output class (options: %s)" % ', '.join(class_options.keys()), metavar="Class")
# process options
args=parser.parse_args()
infile=args.infile
class_name = args.c

if class_name in class_options:
    traind = class_options[class_name]
else:
    print('available classes:')
    for key, val in class_options.iteritems():
        print(key)
    raise Exception('wrong class selection')


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

tdnew = traind()
tdnew._store(x,y,w)
tdnew.writeToFile(outfile+".djctd")
