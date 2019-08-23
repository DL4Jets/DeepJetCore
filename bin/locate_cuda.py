#!/bin/env python

import os
import subprocess

def locate_lib(libname, ldlibpath):
    result=''
    try:
        result = subprocess.check_output(['locate', libname])
        result = result.split('\n')
        res=result[0]
        for l in result:
            if '/usr/' in l:
                res=l
                break
            
        result=os.path.dirname(res)
    except:
        pass
    if len(result) and not (result in ldlibpath):
        if len(ldlibpath):
            ldlibpath+=':'+result+'/'
        else:
            ldlibpath=result
    return ldlibpath    


def locate_cuda():
    ldlibpath = str(os.environ['LD_LIBRARY_PATH'])
    ldlibpath=locate_lib('libcublas.so.10.0',ldlibpath)
    ldlibpath=locate_lib('libcublas.so.9.',ldlibpath)
    ldlibpath=locate_lib('libcudnn.so.7.6',ldlibpath)
    ldlibpath=locate_lib('libcudnn.so.7.5',ldlibpath)
    ldlibpath=locate_lib('libcudnn.so.7.4',ldlibpath)
    ldlibpath=locate_lib('libcusolver.so.10',ldlibpath)
    print(ldlibpath)
    
 
    
locate_cuda()
