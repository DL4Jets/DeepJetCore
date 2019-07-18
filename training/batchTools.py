
from __future__ import print_function

#can pipe config from stdin to condor_submit!
#executable, arguments

#add feedback from traindata arguments
#add not requesting new directory after creating one in the batch submission

from DeepJetCore.training.training_base import training_base
import os, sys, stat

def submit_batch(trainingbase, walltime=None):
    
    subpackage = os.environ['DEEPJETCORE_SUBPACKAGE']
    
    commandline = " ".join(trainingbase.argstring)
    
    scriptpath = trainingbase.outputDir+'batchscript.sh'
    
    condorpath = trainingbase.outputDir+'condor.sub'
    
    #create the batch script
    batch_scipt='''
#!/bin/zsh
echo "running DeepJetCore job::setting up environment"
cd {subpackage}
pwd
source {subpackage}/env.sh
cd {jobdir}
echo "running DeepJetCore job::training"
python {fullcommand}
echo "job done"
    '''.format(subpackage=subpackage, 
               jobdir=trainingbase.outputDir, 
               trainingscript=trainingbase.copied_script,
               fullcommand=commandline+ ' --isbatchrun')
    
    with open(scriptpath,'w') as scriptfile:
        scriptfile.write(batch_scipt)
        
    os.chmod(scriptpath, stat.S_IRWXU |
              stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    
    if walltime is not None:
        wt_days=0
        wt_hours=0
        rest=walltime
        if 'd'  in walltime:
            wt_days = int(rest.split('d')[0])
            rest = rest.split('d')[1:]
        if 'h'  in walltime:
            wt_hours = int(''.join(rest).split('h')[0])
        print('submitting for ', wt_days, 'days', wt_hours, 'hours')
        walltime = wt_days*24*3600 + wt_hours*3600
        
    else:
        walltime=1*24*3600 # 1 day standard
    
    ncpus=3
    ncpus+=trainingbase.ngpus
    
    condor_file='''
executable            = /bin/bash
arguments             = {scriptpath}
output                = {outdir}batch.out
error                 = {outdir}batch.err
log                   = {outdir}batch.log
getenv = True
+MaxRuntime = {walltime}
request_GPUs = {ngpus}
request_cpus = {ncpus}
queue 1
    '''.format(scriptpath=scriptpath,
               outdir=trainingbase.outputDir,
               walltime=str(walltime),
               ngpus=trainingbase.ngpus,
               ncpus=ncpus)
    
    with open(condorpath,'w') as condorfile:
        condorfile.write(condor_file)
    
    os.system('condor_submit '+condorpath)
    print('job submitted')
    
    
    