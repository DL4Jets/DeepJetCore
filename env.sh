

if command -v nvidia-smi > /dev/null
then
        source activate djcenv_gpu
else
        source activate djcenv
fi

export DEEPJETCORE=`pwd`

export PATH=`pwd`/bin:$PATH
export PYTHONPATH=`pwd`/../:$PYTHONPATH
if [ $LD_LIBRARY_PATH ]
then
    export LD_LIBRARY_PATH=`pwd`/compiled/:$LD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=`pwd`/compiled/
fi
export LD_LIBRARY_PATH=`locate_cuda.py`
export LD_PRELOAD=$CONDA_PREFIX/lib/libmkl_core.so:$CONDA_PREFIX/lib/libmkl_sequential.so

