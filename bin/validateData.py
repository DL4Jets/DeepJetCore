#!/usr/bin/env python



from argparse import ArgumentParser
parser = ArgumentParser('Check if all files in a dataset (datacollection) are ok or remove a specific entry\n')
parser.add_argument('inputDataCollection')
parser.add_argument('--remove',default="")
args=parser.parse_args()

from DeepJetCore.DataCollection import DataCollection

dc=DataCollection(args.inputDataCollection)
dc.writeToFile(args.inputDataCollection+".backup")
print('total size before: '+str(dc.nsamples))
if not len(args.remove):
    dc.validate(remove=True)
else:
    dc.removeEntry(args.remove)
    print('total size after: '+str(dc.nsamples))

dc.writeToFile(args.inputDataCollection)