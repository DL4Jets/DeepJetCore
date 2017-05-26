#! /bin/env python

from argparse import ArgumentParser
from pdb import set_trace
import subprocess
import os

parser = ArgumentParser('program to convert root tuples to traindata format')
parser.add_argument("infile", help="set input sample description (output from the check.py script)", metavar="FILE")
parser.add_argument("nchunks", type=int, help="number of jobs to be submitted")
parser.add_argument("out", help="output path")
parser.add_argument("batch_dir", help="batch directory")
parser.add_argument("-c", help="output class")
parser.add_argument("--testdatafor", default='')
parser.add_argument("--nomeans", action='store_true', help='where to get means/std, in case already computed')
args = parser.parse_args()

deep_jet_base = [i for i in os.environ['PYTHONPATH'].split(':') if 'DeepJet' in i]
if len(deep_jet_base) != 1:
   raise RuntimeError('I cannot find the project root directory')
deep_jet_base = os.path.realpath(deep_jet_base[0].split('environment')[0])

proc = subprocess.Popen(
   'voms-proxy-info', 
   stdout=subprocess.PIPE, 
   stderr=subprocess.PIPE
)
if proc.wait() <> 0:
   print "You should have a valid grid proxy to run this!"
   exit()

if not os.path.isdir(args.batch_dir):
   os.mkdir(args.batch_dir)

if not os.path.isdir('%s/batch' % args.batch_dir):
   os.mkdir('%s/batch' % args.batch_dir)

if not (args.nomeans or args.testdatafor):
   #Run a fisrt round of root conversion to get the means/std and weights
   cmd = [
      './convertFromRoot.py', 
      '-i', args.infile,
      '-c', args.c, 
      '-o', args.out, 
      '--means'
      ]
   proc  = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
   out, err = proc.communicate()
   code = proc.wait()
   
   if code != 0:
      raise RuntimeError('The first round of root conversion failed with message: \n\n%s' % err)


inputs = [i for i in open(args.infile)]

def chunkify(l, n):
   """Yield successive n-sized chunks from l."""
   for i in range(0, len(l), n):
      yield l[i:i + n]

if not args.infile.endswith('.txt'):
   raise ValueError('The code assumes that the input files has .txt extension')

txt_template = args.infile.replace('.txt', '.%s.txt')
batch_txts = []
nchunks = 0
for idx, chunk in enumerate(chunkify(inputs, len(inputs)/args.nchunks)):
   name = txt_template % idx
   batch_txts.append(name)
   with open(name, 'w') as cfile:
      cfile.write(''.join(chunk))
   nchunks = idx

batch_template = '''#!/bin/bash
sleep $(shuf -i1-600 -n1) #sleep a random amount of time between 1s and 10' to avoid bottlenecks in reaching afs
echo "JOBSUB::RUN job running"
trap "echo JOBSUB::FAIL job killed" SIGTERM
cd {DJ}/environment/
source lxplus_env.sh
cd {DJ}/convertFromRoot/
./convertFromRoot.py "$@"
exitstatus=$?
if [ $exitstatus != 0 ]
then
echo JOBSUB::FAIL job failed with status $exitstatus
else
echo JOBSUB::SUCC job ended sucessfully
fi
'''.format(DJ=deep_jet_base)
batch_script = '%s/batch.sh' % args.batch_dir
with open(batch_script, 'w') as bb:
   bb.write(batch_template)

means_file = '%s/batch_template.dc' % os.path.realpath(args.out) if not args.testdatafor else args.testdatafor
option = '--usemeansfrom' if not args.testdatafor else '--testdatafor'
with open('%s/submit.sub' % args.batch_dir, 'w') as bb:
   bb.write('''
executable            = {EXE}
arguments             = -i {INFILE} -c {CLASS} -o {OUT} --nothreads --batch conversion.$(ProcId).dc {OPTION} {MEANS}
output                = batch/con_out.$(ProcId).out
error                 = batch/con_out.$(ProcId).err
log                   = batch/con_out.$(ProcId).log
send_credential       = True
getenv = True
use_x509userproxy = True
queue {NJOBS}
'''.format(
   EXE = os.path.realpath(batch_script),
   NJOBS = nchunks,
   INFILE = txt_template % '$(ProcId)',
   CLASS = args.c,
   OUT = os.path.realpath(args.out),
   OPTION = option,
   MEANS = means_file,
)
   )
