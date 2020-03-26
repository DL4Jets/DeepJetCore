#!/usr/bin/env python3



from argparse import ArgumentParser
parser = ArgumentParser('Check if all files in a dataset (datacollection) are ok or remove a specific entry\n')
parser.add_argument('inputDataCollection')
parser.add_argument('--remove',default="")
parser.add_argument('--skip_first',default=0)
args=parser.parse_args()

from DeepJetCore.DataCollection import DataCollection

dc=DataCollection(args.inputDataCollection)
dc.writeToFile(args.inputDataCollection+".backup")

if not len(args.remove):
    dc.validate(remove=True, skip_first=int(args.skip_first))
else:
    dc.removeEntry(args.remove)
    print('total size after: '+str(dc.nsamples))

dc.writeToFile(args.inputDataCollection)