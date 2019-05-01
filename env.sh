

if command -v nvidia-smi > /dev/null
then
        source activate djcenv_gpu
else
        source activate djcenv
fi

export DEEPJETCORE=`pwd`

export PYTHONPATH=`pwd`/../:$PYTHONPATH
export LD_LIBRARY_PATH=`pwd`/compiled:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/patatrack/cuda-9.1/targets/x86_64-linux/lib/:/data/patatrack/cuda-9.0/targets/x86_64-linux/lib/
export PATH=`pwd`/bin:$PATH
export LD_PRELOAD=$CONDA_PREFIX/lib/libmkl_core.so:$CONDA_PREFIX/lib/libmkl_sequential.so
