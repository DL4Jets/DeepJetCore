#! /bin/env python

from argparse import ArgumentParser
from pdb import set_trace
import subprocess
import glob
import os

def grep(fname, pattern):
   with open(fname) as infile:
      for line in infile:
         if pattern in line:
            return True
   return False

parser = ArgumentParser('program to check batch conversion of root tuples to traindata format')
parser.add_argument("indir", help="input dir of the batch task", metavar="FILE")
args = parser.parse_args()

sub_lines = [i for i in open('%s/submit.sub' % args.indir)]
general_lines = []
proc_lines = []
for line in sub_lines:
   if '$(ProcId)' in line:
      proc_lines.append(line.replace('$(ProcId)', '{IDX}'))
   elif 'queue ' in line:
      pass #nothing to be done
   else:
      general_lines.append(line)
proc_lines.append('queue\n')
proc_lines = ''.join(proc_lines)

outputs = glob.glob('%s/batch/con_out.*.out' % args.indir)
if len(outputs)<1:
    print('no jobs submitted, please check')
    exit(-1)

failed = [i for i in outputs if not grep(i, 'JOBSUB::SUCC')]
successful=[  i.split(".")[-2]   for i in outputs if grep(i, 'JOBSUB::SUCC')]

def get_output_dir():
   batch_args = [i for i in open('%s/submit.sub' % args.indir) if 'arguments' in i][0]
   batch_args = batch_args.split('=')[1].split('-')
   output_dir = [i for i in batch_args if i.startswith('o ')][0].split(' ')[1]
   return output_dir



def merge_successful():
    output_dir=get_output_dir()
    from DeepJetCore.DataCollection import DataCollection
    alldc=[]
    for s in successful:
        in_path=output_dir+'/conversion.'+str(s)+'.dc'
        alldc.append(DataCollection(in_path))
    print("merging DataCollections")
    merged = sum(alldc)
    print("saving merged DataCollection")
    merged.writeToFile('%s/dataCollection.dc' % output_dir)
    print('successfully merged to %s/dataCollection.dc' % output_dir)
    return merged
    

if len(failed) == 0:
   print 'All jobs successfully completed, merging...'
   merged=merge_successful()
   dname = os.path.dirname(merged.originRoots[0])
   infiles = glob('%s/*.root' % dname)
   if len(infiles) != len(merged.originRoots):
      print '\n\n\nThere are missing files that were not converted, maybe something went wrong!\n\n\n'
   
else:
   keep_going = raw_input('%d/%d jobs have failed, should I recover them? [yY/nN]   ' % (len(failed), len(outputs)))
   if keep_going.lower() == 'n': 
       merge_anyways = raw_input('Should I merge the sucessfully converted files (%d)? [yY/nN]    ' % len(successful))
       if merge_anyways.lower() == 'n': exit(0)
       merge_successful()
       exit(0)

   idxs = [os.path.basename(i).split('.')[1] for i in failed]
   with open('%s/rescue.sub' % args.indir, 'w') as jdl:
      jdl.write(''.join(general_lines))   
      jdl.write('\n'.join([proc_lines.format(IDX=i) for i in idxs]))
   print 'rescue file created'
