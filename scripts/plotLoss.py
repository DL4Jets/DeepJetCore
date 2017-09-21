#!/usr/bin/env python

import matplotlib

matplotlib.use('Agg')     


from argparse import ArgumentParser
parser = ArgumentParser('')
parser.add_argument('inputDir')
parser.add_argument("--file",  help="specify loss file", metavar="FILE", default='losses.log')
parser.add_argument("--range",  help="specify y axis range",  nargs='+', type=float, metavar="OPT", default=[])
    
args = parser.parse_args()

infilename=args.inputDir+'/'+args.file

infile=open(infilename,'r')

trainloss=[]
valloss=[]
epochs=[]
i=0
automax=0
automin=100
for line in infile:
    if len(line)<1: continue
    tl=float(line.split(' ')[0])
    vl=float(line.split(' ')[1])
    trainloss.append(tl)
    valloss.append(vl)
    epochs.append(i)
    i=i+1
    if i==5:
        automax=max(tl,vl)
    automin=min(automin,vl,tl)
    

import matplotlib.pyplot as plt
f = plt.figure()
plt.plot(epochs,trainloss,'r',epochs,valloss,'b')
plt.ylabel('loss')
plt.xlabel('epoch')
if len(args.range)==2:
    plt.ylim(args.range)
elif automax>0:
    plt.ylim([automin*0.9,automax])
f.savefig(args.inputDir+'/'+"losses.pdf")

