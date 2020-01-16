    
from DeepJetCore.DJCLosses import *
from DeepJetCore.DJCLayers import *
import imp
try:
    imp.find_module('Losses')
    from Losses import *
except ImportError:
    print ('No Losses module found, ignoring at your own risk')
    global_loss_list = {}

try:
    imp.find_module('Layers')
    from Layers import *
except ImportError:
    print ('No Layers module found, ignoring at your own risk')
    global_layers_list = {}

try:
    imp.find_module('Metrics')
    from Metrics import *
except ImportError:
    print ('No metrics module found, ignoring at your own risk')
    global_metrics_list = {}    

def get_custom_objects():
    
    custom_objs = {}
    custom_objs.update(djc_global_loss_list)
    custom_objs.update(djc_global_layers_list)
    custom_objs.update(global_loss_list)
    custom_objs.update(global_layers_list)
    custom_objs.update(global_metrics_list)
    return custom_objs
