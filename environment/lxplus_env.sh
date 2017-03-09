

source activate deepjetLinux
export PYTHONPATH=`pwd`/../modules:$PYTHONPATH
export LD_LIBRARY_PATH=`pwd`/../modules:$LD_LIBRARY_PATH

#to avoid stack overflow due to very large python arrays
ulimit -s 65532
