

source activate deepjetLinux3_gpu

export PYTHONPATH=`pwd`/../modules:$PYTHONPATH
export LD_LIBRARY_PATH=`pwd`/../modules:$LD_LIBRARY_PATH
export PATH=`pwd`/../scripts:$PATH
export LD_PRELOAD=$CONDA_PREFIX/lib/libmkl_core.so:$CONDA_PREFIX/lib/libmkl_sequential.so
