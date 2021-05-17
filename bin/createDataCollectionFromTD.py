#!/usr/bin/env python3
# encoding: utf-8


from DeepJetCore.DataCollection import DataCollection
from DeepJetCore.conversion.conversion import class_options
from DeepJetCore.TrainData import TrainData
from argparse import ArgumentParser
import os

parser = ArgumentParser('program to wrap converted trainData files in a dataCollection and attach a python TrainData class description')
parser.add_argument("-c",  choices = class_options.keys(), help="set output class (options: %s)" % ', '.join(class_options.keys()), metavar="Class")
parser.add_argument("-o",  help="dataCollection output file name",default="")

parser.add_argument('files', metavar='N',nargs='+',
                    help='djctd files to be merged in the DataCollection')

args=parser.parse_args()


if len(args.files) < 1:
    print('you must provide at least one input file')
    exit()
if not len(args.o):
    print('you must provide an output file name')
    exit()

indir = os.path.dirname(args.files[0])
if len(indir):
    indir+="/"
class_name = args.c

if class_name in class_options:
    traind = class_options[class_name]
else:
    print('available classes:')
    for key, val in class_options.items():
        print(key)
    raise Exception('wrong class selection')

dc = DataCollection()
dc.setDataClass(traind)

for f in args.files:
   dc.samples.append(os.path.basename(f))

outfile = args.o
if not outfile[-6:] == ".djcdc":
    outfile+=".djcdc"
dc.writeToFile(indir+outfile)
