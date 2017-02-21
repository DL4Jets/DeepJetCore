Help to setup the environment on your machine

The code is test and run using package management with anaconda:
https://www.continuum.io/anaconda-overview

If you installed anaconda, you can use the .yml file to install the version we used. 
The .sh is a small macro that does the installation and environment definition used 
for testing deepJet (on the laptops).
 

Each time before running, the environment should be activated and the PYTHONPATH needs to be adapted.
This can be easily done for zsh/bash/sh shells with 

source env.sh

The script needs to be called from this directory