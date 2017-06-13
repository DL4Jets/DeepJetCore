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

parser = ArgumentParser('program to convert root tuples to traindata format')
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
failed = [i for i in outputs if not grep(i, 'JOBSUB::SUCC')]

if len(failed) == 0:
   print 'All jobs successfully completed, merging...'
   from DataCollection import DataCollection
   from glob import glob
   batch_args = [i for i in open('%s/submit.sub' % args.indir) if 'arguments' in i][0]
   batch_args = batch_args.split('=')[1].split('-')
   output_dir = [i for i in batch_args if i.startswith('o ')][0].split(' ')[1]
   merged = sum(DataCollection(i) for i in glob('%s/conversion.*.dc' % output_dir))
   dname = os.path.dirname(merged.originRoots[0])
   infiles = glob('%s/*.root' % dname)
   if len(infiles) != len(merged.originRoots):
      print '\n\n\nThere are missing files that were not converted, maybe something went wrong!\n\n\n'
   merged.writeToFile('%s/dataCollection.dc' % output_dir)
else:
   keep_going = raw_input('%d/%d jobs have failed, should I recover them? [yY/nN]   ' % (len(failed), len(outputs)))
   if keep_going.lower() == 'n': exit(0)

   idxs = [os.path.basename(i).split('.')[1] for i in failed]
   with open('%s/rescue.sub' % args.indir, 'w') as jdl:
      jdl.write(''.join(general_lines))   
      jdl.write('\n'.join([proc_lines.format(IDX=i) for i in idxs]))
   print 'rescue file created'
