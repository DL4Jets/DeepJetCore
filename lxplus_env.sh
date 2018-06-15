
# Assume user has already activated conda environment
# conda activate deepjetenv

# export DEEPJETCORE=`pwd`

export PYTHONPATH=`pwd`/../:`pwd`:$PYTHONPATH
export LD_LIBRARY_PATH=`pwd`/compiled:$LD_LIBRARY_PATH
export PATH=`pwd`/bin:$PATH
export LD_PRELOAD=$CONDA_PREFIX/lib/libmkl_core.so:$CONDA_PREFIX/lib/libmkl_sequential.so:$LD_PRELOAD


#to avoid stack overflow due to very large python arrays
ulimit -s 65532
