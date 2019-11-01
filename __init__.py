
import imp
try:
    imp.find_module('setGPU')
    import setGPU
except:
    pass
        
import sys
import tensorflow as tf
sys.modules["keras"] = tf.keras

import matplotlib
matplotlib.use('Agg')
