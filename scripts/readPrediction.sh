
#!/bin/bash



export ROOT_TREEASSOCIATIONINFILE=$1
export CPLUS_INCLUDE_PATH=$DEEPJET/modules/interface
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib64/root
root -l  $DEEPJET/scripts/loadTreeAssociation.C