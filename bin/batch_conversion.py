#!/bin/env python3

import os
import logging
from argparse import ArgumentParser

logging.getLogger().setLevel(logging.INFO)

parser = ArgumentParser('program to convert root tuples to traindata format')
parser.add_argument("infile", help="set input sample description (output from the check.py script)", metavar="FILE")
parser.add_argument("nchunks", type=int, help="number of jobs to be submitted")
parser.add_argument("out", help="output path")
parser.add_argument("batch_dir", help="batch directory")
parser.add_argument("-c", help="output class", default="")
parser.add_argument("--classArgs",  help="Arguments to pass to output class")
parser.add_argument("--testdatafor", default='')
parser.add_argument("--nforweighter", default='500000', help='set number of samples to be used for weight and mean calculation')
parser.add_argument("--meansfrom", default="", help='where to get means/std, in case already computed')
parser.add_argument("--useexistingsplit", default=False, help='use an existing file split (potentially dangerous)')
parser.add_argument("--noRelativePaths", help="Assume input samples are absolute paths with respect to working directory", default=False, action="store_true")
parser.add_argument("--jobFlavour", default='longlunch', help="CERN HTCondor job flavour (espresso, microcentury, longlunch, workday)")
parser.add_argument("--cmst3", action="store_true", help="Submit jobs with cmst3 accounting group.")
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

    from DeepJetCore.DataCollection import DataCollection
    from DeepJetCore.conversion.conversion import class_options

    try:
        cls = class_options[args.c]
    except KeyError:
        raise Exception('wrong class selection')

    if not args.classArgs:
        args.classArgs = tuple()

    dc = DataCollection(nprocs=-1)
    dc.meansnormslimit = int(args.nforweighter)
    try:
        dc.convertListOfRootFiles(args.infile, cls(*args.classArgs), args.out,
                                  means_only=True,
                                  output_name='batch_template.dc',
                                  relpath=('' if args.noRelativePaths else os.path.dirname(os.path.realpath(args.infile)))
        )
    
    except:
        print 'The first round of root conversion failed'
        raise

    print('means/norms/weighter produced successfully')

elif args.meansfrom:
    if not os.path.exists(args.meansfrom):
        raise Exception("The file "+args.meansfrom+" does not exist")
    print('using means/weighter from '+args.meansfrom)
    os.mkdir(args.out)
    os.system('cp '+args.meansfrom+' '+args.out+'/batch_template.dc')

if not args.infile.endswith('.txt'):
    raise ValueError('The code assumes that the input files has .txt extension')

with open(args.infile) as source:
    num_inputs = len(source.read().split('\n'))

chunk_size = num_inputs / args.nchunks

print('splitting input file...')
range_indices = []

for idx, start in enumerate(range(0, num_inputs, chunk_size)):
    range_indices.append((idx, start, start + chunk_size))

batch_template = '''#!/bin/bash
#sleep $(shuf -i1-300 -n1) #sleep a random amount of time between 1s and 10' to avoid bottlenecks in reaching afs
echo "JOBSUB::RUN job running"
trap "echo JOBSUB::FAIL job killed" SIGTERM
BASEDIR=`pwd`
cd {subpackage}
source env.sh
convertFromSource.py "$@"
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

options = []
if args.noRelativePaths:
    options.append('--noRelativePaths')
if args.testdatafor:
    options.append('--testdatafor ' + args.testdatafor)
else:
    options.append('--usemeansfrom %s/batch_template.dc' % os.path.realpath(args.out))

option = ' '.join(options)

with open('%s/submit.sub' % args.batch_dir, 'w') as bb:
    bb.write('''executable            = {EXE}
arguments             = -i {INFILE} --inRange $(START) $(STOP) -c {CLASS} -o {OUT} --nothreads --batch conversion.$(JOBIDX).dc {OPTION}
output                = {BATCH_DIR}/batch/con_out.$(JOBIDX).out
error                 = {BATCH_DIR}/batch/con_out.$(JOBIDX).err
log                   = {BATCH_DIR}/batch/con_out.$(JOBIDX).log
#+MaxRuntime = 86399
+JobFlavour = "{FLAVOUR}"
getenv = True
#use_x509userproxy = True
accounting_group = {ACCTGRP}
+AccountingGroup = {ACCTGRP}   
queue JOBIDX START STOP from (
{RANGE_INDICES}
)
'''.format(
    EXE = os.path.realpath(batch_script),
    INFILE = args.infile,
    CLASS = args.c,
    OUT = os.path.realpath(args.out),
    OPTION = option,
    BATCH_DIR = args.batch_dir,
    FLAVOUR = args.jobFlavour,
    ACCTGRP = 'group_u_CMST3.all' if args.cmst3 else 'group_u_CMS.u_zh',
    RANGE_INDICES = '\n'.join('%d %d %d' % rng for rng in range_indices)
))
   
print('condor submit file can be found in '+ args.batch_dir+'\nuse check_conversion.py ' + args.batch_dir + ' to to check jobs')
