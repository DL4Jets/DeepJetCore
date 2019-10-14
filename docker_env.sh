export DEEPJETCORE=`pwd`
export PATH=`pwd`/bin:$PATH
export PYTHONPATH=`pwd`/../:$PYTHONPATH
if [ $LD_LIBRARY_PATH ]
then
    export LD_LIBRARY_PATH=`pwd`/compiled/:$LD_LIBRARY_PATH
else
    export LD_LIBRARY_PATH=`pwd`/compiled/
fi
