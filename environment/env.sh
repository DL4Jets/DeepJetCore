

source activate deepjet
export PYTHONPATH=`pwd`/../modules:$PYTHONPATH

#to avoid stack overflow due to very large python arrays
ulimit -s `ulimit -s -H`