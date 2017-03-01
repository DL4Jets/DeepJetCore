

source activate deepjetLinux
export PYTHONPATH=`pwd`/../modules:$PYTHONPATH

#to avoid stack overflow due to very large python arrays
ulimit -s 65532
