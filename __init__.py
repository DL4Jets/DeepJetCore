
import imp
try:
    imp.find_module('setGPU')
    import setGPU
except:
    pass
        
import sys
import tensorflow 
#for now let's keep it compatible

#maybe we can leave this switched on
#tensorflow.compat.v1.disable_eager_execution()
#sys.modules["tensorflow"]=tensorflow.compat.v1

sys.modules["keras"] = tensorflow.keras

#no X in current containers
import matplotlib
matplotlib.use('Agg')
