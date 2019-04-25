#activate deepjetLinux3_gpu if no other conda env is already active
if [ -z "$CONDA_PREFIX" ]
then
		source activate deepjetLinux3
fi

export DEEPJETCORE=`pwd`

export PYTHONPATH=`pwd`/../:$PYTHONPATH
export LD_LIBRARY_PATH=`pwd`/compiled:$LD_LIBRARY_PATH
export PATH=`pwd`/bin:$PATH
export LD_PRELOAD=$CONDA_PREFIX/lib/libmkl_core.so:$CONDA_PREFIX/lib/libmkl_sequential.so


#to avoid stack overflow due to very large python arrays
ulimit -s 65532
