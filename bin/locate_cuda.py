#!/bin/env python

import os
import subprocess


def locate_cuda():
    ldlibpath = str(os.environ['LD_LIBRARY_PATH'])
    result=''
    try:
        result = os.path.dirname(subprocess.check_output(['locate', 'libcublas.so.9.0']).split('\n')[0])
    except:
        pass
    if not (result in ldlibpath):
        ldlibpath+=':'+result+'/'
    result=''
    try:
        result = os.path.dirname(subprocess.check_output(['locate', 'libcudnn.so.7']).split('\n')[0])
    except: 
        pass
    if not (result in ldlibpath):
        ldlibpath+=':'+result+'/'
    print(ldlibpath)
    
    
locate_cuda()