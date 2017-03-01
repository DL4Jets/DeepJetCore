Help to setup the environment on your machine

On the gpu, only do:

source gpu_env.sh
If you have installed miniconda on lxplus, you need to remove its directory from your PATH environment!


On Lxplus/Mac, you need to install miniconda in your workspace [see 1]. 

After that it is sufficient to run (zsh, bash, sh):

source env.sh (Mac)
source lxplus_env.sh (Lxplus)





[1]
The code is test and run using package management with anaconda or miniconda:
https://www.continuum.io/anaconda-overview
On lxplus, miniconda is recommended, since it needs less disk space!


If you installed anaconda/miniconda, you can use the .yml file to install the version we used. 
The setupEnv.sh is a small macro that does the installation and environment definition.
Please call:

bash setupEnv.sh conda_deepjet.yml (on Mac)
bash setupEnv.sh conda_deepjetLinux.yml (on lxplus)

Each time before running, the environment should be activated and the PYTHONPATH needs to be adapted.
This can be easily done for zsh/bash/sh shells with 

source env.sh (Mac)
source lxplus_env (Linux)

The script needs to be called from this directory