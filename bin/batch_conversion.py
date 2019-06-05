#!/bin/env python

from argparse import ArgumentParser
from pdb import set_trace
import subprocess
import os

parser = ArgumentParser('program to convert root tuples to traindata format')
parser.add_argument("infile", help="set input sample description (output from the check.py script)", metavar="FILE")
parser.add_argument("nchunks", type=int, help="number of jobs to be submitted")
parser.add_argument("out", help="output path")
parser.add_argument("batch_dir", help="batch directory")
parser.add_argument("-c", help="output class", default="")
parser.add_argument("--testdatafor", default='')
parser.add_argument("--nforweighter", default='500000', help='set number of samples to be used for weight and mean calculation')
parser.add_argument("--meansfrom", default="", help='where to get means/std, in case already computed')
parser.add_argument("--useexistingsplit", default=False, help='use an existing file split (potentially dangerous)')
args = parser.parse_args()

args.infile = os.path.abspath(args.infile)
args.out = os.path.abspath(args.out)
args.batch_dir = os.path.abspath(args.batch_dir)

if len(args.c)<1:
    print("please specify and output class")
    exit(-1)


deep_jet_base = os.environ['DEEPJETCORE_SUBPACKAGE']
if len(deep_jet_base) < 1:
   raise RuntimeError('I cannot find the project root directory. DEEPJETCORE_SUBPACKAGE needs to be defined')


deep_jet_base_name = os.path.basename(deep_jet_base)
deep_jet_core  = os.path.abspath((os.environ['DEEPJETCORE']))


if os.path.isdir(args.out):
    print ("output dir must not exists")
    exit(-2)

if os.path.isdir(args.batch_dir):
    print ("batch dir must not exists")
os.mkdir(args.batch_dir)

if not os.path.isdir('%s/batch' % args.batch_dir):
    os.mkdir('%s/batch' % args.batch_dir)   
   

if not (len(args.meansfrom) or args.testdatafor):
    #Run a fisrt round of root conversion to get the means/std and weights
    print('creating a dummy datacollection for means/norms and weighter (can take a while)...')
    cmd = [
       'convertFromRoot.py', 
       '-i', args.infile,
       '-c', args.c, 
       '-o', args.out, 
       '--nforweighter', args.nforweighter,
       '--means'
       ]
    proc  = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    code = proc.wait()
    
    if code != 0:
        raise RuntimeError('The first round of root conversion failed with message: \n\n%s' % err)
    else:
        print('means/norms/weighter produced successfully')

elif args.meansfrom:
    if not os.path.exists(args.meansfrom):
        raise Exception("The file "+args.meansfrom+" does not exist")
    print('using means/weighter from '+args.meansfrom)
    os.mkdir(args.out)
    os.system('cp '+args.meansfrom+' '+args.out+'/batch_template.dc')

inputs = [i for i in open(args.infile)]

def chunkify(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

if not args.infile.endswith('.txt'):
    raise ValueError('The code assumes that the input files has .txt extension')


print('splitting input file...')
txt_template = args.infile.replace('.txt', '.%s.txt')
batch_txts = []
nchunks = 0
for idx, chunk in enumerate(chunkify(inputs, len(inputs)/args.nchunks)):
    name = txt_template % idx
    batch_txts.append(name)
    if not args.useexistingsplit:
        with open(name, 'w') as cfile:
            cfile.write(''.join(chunk))
    nchunks = idx


batch_template = '''#!/bin/bash
sleep $(shuf -i1-300 -n1) #sleep a random amount of time between 1s and 10' to avoid bottlenecks in reaching afs
echo "JOBSUB::RUN job running"
trap "echo JOBSUB::FAIL job killed" SIGTERM
BASEDIR=`pwd`
cd {subpackage}
source env.sh
convertFromRoot.py "$@"
exitstatus=$?
if [ $exitstatus != 0 ]
then
echo JOBSUB::FAIL job failed with status $exitstatus
else
echo JOBSUB::SUCC job ended sucessfully
fi
'''.format(subpackage=deep_jet_base)
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
+MaxRuntime = 86399
getenv = True
use_x509userproxy = True
queue {NJOBS}
'''.format(
   EXE = os.path.realpath(batch_script),
   NJOBS = nchunks+1,
   INFILE = txt_template % '$(ProcId)',
   CLASS = args.c,
   OUT = os.path.realpath(args.out),
   OPTION = option,
   MEANS = means_file,
)
   )
   
print('condor submit file can be found in '+ args.batch_dir+'\nuse check_conversion.py ' + args.batch_dir + ' to to check jobs')
