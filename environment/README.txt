Help to setup the environment on your machine

On the gpu, only do:

source gpu_env.sh
If you have installed miniconda on lxplus, you need to remove its directory from your PATH environment!


On Lxplus/Mac, you need to install miniconda in your workspace see [1]. 

After that it is sufficient to run (zsh, bash, sh):

source env.sh (Mac)
source lxplus_env.sh (Lxplus)

In addition, the compiled modules need to be compiled. 
This cannot be done directly on the GPU machines. please compile on lxplus and use the compiled libraries on the GPU
After sourcing the environment scripts, run 'make' in the 'modules' directory.



[1]
The code is tested and run using package management with anaconda or miniconda:
https://www.continuum.io/anaconda-overview
On lxplus, miniconda is recommended, since it needs less disk space!
Please make sure, conda is added to your path (you will be prompted). Answer with "yes" or take care yourself
that the command 'which conda' returns the path of your conda installation before you use the package.

If you installed anaconda/miniconda, you can use the .conda file to install the version we used. 
The setupEnv.sh is a small macro that does the installation and environment definition.
Please call:

 ./setupEnv.sh deepjetLinux3.conda 

Each time before running, the environment should be activated and the PYTHONPATH needs to be adapted.
This can be easily done for zsh/bash/sh shells with 

source env.sh (Mac)
source lxplus_env (Linux)

The script needs to be called from this directory