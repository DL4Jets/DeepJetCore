#!/usr/bin/env python
    
from DeepJetCore.evaluation import plotLoss


from argparse import ArgumentParser
parser = ArgumentParser('')
parser.add_argument('inputDir')
parser.add_argument("--file",  help="specify loss file", metavar="FILE", default='losses.log')
parser.add_argument("--range",  help="specify y axis range",  nargs='+', type=float, metavar="OPT", default=[])
    
args = parser.parse_args()

infilename=args.inputDir+'/'+args.file



plotLoss(infilename,args.inputDir+'/losses.pdf',args.range)
