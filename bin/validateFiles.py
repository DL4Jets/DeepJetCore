#!/usr/bin/env python3


from argparse import ArgumentParser
import os
from DeepJetCore.conversion.conversion import class_options
import tqdm

parser = ArgumentParser('Check if all files in a file list and remove broken entries\n')
parser.add_argument('inputFileList')
parser.add_argument("-c",  choices = class_options.keys(), help="set output class (required, options: %s)" % ', '.join(class_options.keys()), metavar="Class")

args=parser.parse_args()

class_name=args.c

traind = None
if class_name in class_options:
    traind = class_options[class_name]()
else:
    print('available classes:')
    for key, val in class_options.items():
        print(key)
    exit()
    
infiles = []
inputdir = os.path.abspath(os.path.dirname(args.inputFileList))
if len(inputdir):
    inputdir+="/"
    
with open(args.inputFileList, "r") as f:
    for s in f:
        if len(s):
            infiles.append(s[:-1])#remove '\n'
            
os.system("cp -f "+args.inputFileList+" "+args.inputFileList+".backup")

removedfiles=[]
with open(args.inputFileList, "w") as f:
    for s in tqdm.tqdm(infiles):
        if traind.fileIsValid(inputdir+s):
            f.write(s+'\n')
        else:
            removedfiles.append(inputdir+s)
            
print('files removed',removedfiles)