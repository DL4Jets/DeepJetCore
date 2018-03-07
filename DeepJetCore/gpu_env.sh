

source activate deepjetLinux3_gpu

export DEEPJETCORE=`pwd`

export PYTHONPATH=`pwd`/../:$PYTHONPATH
export LD_LIBRARY_PATH=`pwd`/compiled:$LD_LIBRARY_PATH
export PATH=`pwd`/bin:$PATH
export LD_PRELOAD=$CONDA_PREFIX/lib/libmkl_core.so:$CONDA_PREFIX/lib/libmkl_sequential.so
