## Installation

* While installing on lxplus7 (based on CentOS 7.5), these are the steps you can follow. Ensure you have the correct OS before proceeding.

#### Anaconda Setup

* Note: This will require disk space, especially if you work with multiple conda environments so ensure you have enough disk space (at least 12 GB).

```
    $ mkdir <new-directory> 
    $ cd <new-directory>
    $ wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    $ bash Miniconda3-latest-Linux-x86_64.sh
```

* Once Anaconda is setup you can work with virtual environments as and when you like, cloning or creating new ones. In this case, we clone an environment from a file in order to use DeepJet on lxplus7.

```
    $ wget https://raw.githubusercontent.com/SwapneelM/DeepJetCore/python-package/DeepJetCore/environment/djlt.yaml
    $ conda env create -f djlt.yaml -n deepjetpkg
```

* In case install fails, and you want to remove the environment then use:

```   
    $ conda env remove -n deepjetpkg
```

*Conda automatically allows you to use `source activate <env_name>` for environments but it would be better if you follow the conda installation instructions to enable the command `conda` by setting the conda path in your ~/.bashrc file*

```
    $ echo ". path/to/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc
    $ source ~/.bashrc
```

*Otherwise replace `conda activate/deactivate` with `source activate/deactivate` where necessary and add this as suggested by miniconda during installation of the environment.*

```
    $ echo export PATH="/afs/cern.ch/work/s/smehta/miniconda3/bin:$PATH" >> ~/.bashrc
    $ source ~/.bashrc
```

#### Clone the DeepJet Repository

* DeepJetCore is a set of scripts aimed at providing a supervised learning environment for Physics. A set of examples for understanding usage of DeepJetCore is provided in DeepJet that allows users to understand and add their own architectures and datastructures to retrieve root files and train models on the data.

```
    $ git clone -b python-package https://github.com/SwapneelM/DeepJet
```

* Note: The Python package (DeepJetCore==0.0.5) is added as a dependency in the environment file itself so you should not have to install it separately, but in case PyPi is slow (>30s)/fails in downloading the package, remove the line with the package from the environment file and re-install the same environment. 
[PyPi isn't working out at the moment so just clone it until the changes propagate to PyPi]

* **Personally, I would recommend you clone the repository to follow what is happening and better understand the functioning of the library.**

* These are the steps to follow to separately install DeepJetCore:

- Stay in the same root directory as the one where you cloned DeepJet.

```
    $ git clone -b python-package https://github.com/SwapneelM/DeepJetCore/
    $ cd DeepJetCore 
    $ conda activate deepjetpkg
    (deepjetpkg) $ python setup.py build install 
```

- This step will take a while as it compiles all the dependencies and figures out the linking of libraries.

#### Activate and Use DeepJet/DeepJetCore

* Currently, you will need to set some environment variables each time you activate the virtual environment which are provided in the file `pypkg_env.sh` 

```
    $ conda activate deepjetpkg
    $ cd DeepJet
   (deepjetpkg) $ source pypkg_env.sh
```

* Now that you have installed the libraries, follow the [README.md](https://github.com/SwapneelM/DeepJet) for DeepJet in order to better understand the instructions and execution of commands within the library.
    
#### Common Errors

* `libstdc++.so.6 : GLIBCXX...` version not found: Your libstdc++.so.6 has probably been symlinked against an older version of libstdc++.so.6 (e.g. libstdc++.so.6.0.19). Recreating this symlink against a newer version (e.g. libstdc++.so.6.0.24) should do the job for you. [This could prove a useful StackOverflow reference point](https://stackoverflow.com/a/16445803/5087991)

```
    $ cd $CONDA_PREFIX/lib
```
  
  - Check which version of libstdc++.so.6 is actually sym-linked and any other versions available with GLIBCXX (here we assume it is libstdc++.so.6.0.24)

```
    (deepjetpkg) $ ls -ltr libstdc++.so.6*
    (deepjetpkg) $ strings libstdc++.so.6.0.* | grep 'GLIBCXX'
    (deepjetpkg) $ ln -sf libstdc++.so.6.0.24 libstdc++.so.6
```

* `'datastructures' submodule not found`: Please check if you have added the `DeepJet/modules` folder to the $PYTHONPATH environment variable.

* `libquicklz.so not found`: Check if you have added DeepJetCore/compiled folder to the path. It is either going to be in `$CONDA_PREFIX/lib/python2.7/site-packages/DeepJetCore(version)/compiled` or if you have cloned DeepJetCore then simply `DeepJetCore/DeepJetCore/compiled`

* Tensorflow 1.9.0 requires setuptools <= 39.1.0 and you might have a different version installed. It can cause errors later so it is probably better to run an install with the requisite version AFTER activating the conda environment.

```
    (deepjetpkg) $ pip install setuptools==39.1.0
```
- Update: This has been added to the environment file so if you face this issue, you have probably set up an older version of the environment. Just install setuptools as shown above to resolve the error.

* `pkg_resources.ResolutionError: Script 'convertFromRoot.py' not found in scripts/`: This is probably a result of an incorrect/incomplete install of DeepJetCore and can be resolved by uninstalling and reinstalling the package in your conda environment.

```
    (deepjetpkg) $ pip uninstall DeepJetCore # library names ignore case
    (deepjetpkg) $ cd DeepJetCore
    (deepjetpkg) $ python setup.py build install  
```
- If that doesn't work then the current fallback solution is to reinstall miniconda altogether.

* Root library linking errors; undefined symbols: Ensure that `$LD_PRELOAD` and `$LD_LIBRARY_PATH` have been set according to the paths in `pypkg_env.sh` file in DeepJet. Then recompile the shared libs or reinstall DeepJetCore to re-link the shared libs.

**Other errors require different kinds of fixes so open an issue or send me an email and I'll get back to you with a solution.**








