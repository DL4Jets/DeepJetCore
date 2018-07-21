## Installation
===============

* While installing on lxplus7 (based on CentOS 7.5), these are the steps you can follow. Ensure you have the correct OS before proceeding.

#### Anaconda Setup

* Note: This will require disk space, especially if you work with multiple conda environments so ensure you have enough disk space (>40GB).

    $ mkdir <new-directory> 
    $ cd <new-directory>
    $ wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    $ bash Miniconda3-latest-Linux-x86_64.sh

* Once Anaconda is setup you can work with virtual environments as and when you like, cloning or creating new ones. In this case, we clone an environment from a file in order to use DeepJet on lxplus7.

    $ wget https://raw.githubusercontent.com/SwapneelM/DeepJetCore/python-package/DeepJetCore/environment/djlt.yaml
    $ conda create -n deepjetLinuxtest --file djlt.yaml

#### Clone the DeepJet Repository

* DeepJetCore is a set of scripts aimed at providing a supervised learning environment for Physics. A set of examples for understanding usage of DeepJetCore is provided in DeepJet that allows users to understand and add their own architectures and datastructures to retrieve root files and train models on the data.

    $ git clone -b python-package https://github.com/SwapneelM/DeepJet

* Note: The Python package (DeepJetCore==0.0.5) is added as a dependency in the environment file itself so you should not have to install it separately, but in case PyPi is slow (>30s)/fails in downloading the package, remove the line with the package from the environment file and re-install the same environment. 

* **Personally, I would recommend you clone the repository to follow what is happening and better understand the functioning of the library.**

* After that, these are the steps to follow to separately install DeepJetCore:

* Stay in the same root directory as the one where you cloned DeepJet.

    $ git clone -b python-package https://github.com/SwapneelM/DeepJetCore/
    $ cd DeepJetCore 
    $ conda activate deepjetLinuxtest
    $ python setup.py build install 

* This step will take a while as it compiles all the dependencies and figures out the linking of libraries.

#### Activate and Use DeepJet/DeepJetCore

* Currently, you will need to set some environment variables each time you activate the virtual environment which are provided in the file `pypkg_env.sh` 

    $ conda activate deepjetLinuxtest
    $ cd DeepJet
    $ source pypkg_env.sh

* Now that you have installed the libraries, follow the [README.md](https://github.com/SwapneelM/DeepJet) for DeepJet in order to better understand the instructions and execution of commands within the library.
    
#### Common Errors

* `libstdc++.so.6 : GLIBCXX...` version not found: Your libstdc++.so.6 has probably been symlinked against an older version of libstdc++.so.6 (e.g. libstdc++.so.6.0.19). Recreating this symlink should do the job for you. [This could prove a useful StackOverflow reference point](https://stackoverflow.com/a/16445803/5087991)

* `'datastructures' submodule not found`: Please check if you have added the `DeepJet/modules` folder to the $PYTHONPATH environment variable.

* Root linking errors: These vary so send me an email and I'll get back with a solution.

* `libquicklz.so not found`: Check if you have added DeepJetCore/compiled folder to the path. It is either going to be in `$CONDA_PREFIX/lib/python2.7/site-packages/DeepJetCore(version)/compiled` or if you have cloned DeepJetCore then simply `DeepJetCore/DeepJetCore/compiled`









