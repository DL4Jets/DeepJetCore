import imp
try:
    imp.find_module('Losses')
    from Losses import *
except ImportError:
    print 'No Losses module found, ignoring at your own risk'
    global_loss_list = {}

try:
    imp.find_module('Layers')
    from Layers import *
except ImportError:
    print 'No Layers module found, ignoring at your own risk'
    global_layers_list = {}

try:
    imp.find_module('Metrics')
    from Metrics import *
except ImportError:
    print 'No metrics module found, ignoring at your own risk'
    global_metrics_list = {}



custom_objs = {}
custom_objs.update(global_loss_list)
custom_objs.update(global_layers_list)
custom_objs.update(global_metrics_list)
    
    
def getLayer(model, name):
    for layer in model.layers:
        if layer.name == name:
            return layer
        



def printLayerInfosAndWeights(model, noweights=False):
    for layer in model.layers:
        g=layer.get_config()
        h=layer.get_weights()
        print (g)
        if noweights: continue
        print (h)


def fixLayersContaining(m, fixOnlyContaining, invert=False):
    isseq=(not hasattr(fixOnlyContaining, "strip") and
            hasattr(fixOnlyContaining, "__getitem__") or
            hasattr(fixOnlyContaining, "__iter__"))
    if not isseq:
        fixOnlyContaining=[fixOnlyContaining]
    if invert:
        for layidx in range(len(m.layers)):
            m.get_layer(index=layidx).trainable=False
        for layidx in range(len(m.layers)):
            for ident in fixOnlyContaining:
                if len(ident) and ident in m.get_layer(index=layidx).name:
                    m.get_layer(index=layidx).trainable=True
    else:
        for layidx in range(len(m.layers)):
            for ident in fixOnlyContaining:
                if len(ident) and ident in m.get_layer(index=layidx).name:
                    m.get_layer(index=layidx).trainable=False
    return m

def set_trainable(m, patterns, value):
    if isinstance(patterns, basestring):
        patterns = [patterns]
    for layidx in range(len(m.layers)):
        name = m.get_layer(index=layidx).name
        if any(i in name for i in patterns):
            m.get_layer(index=layidx).trainable = value
    return m

def setAllTrainable(m):
    for layidx in range(len(m.layers)):
        m.get_layer(index=layidx).trainable = True
    return m

def loadModelAndFixLayers(filename,fixOnlyContaining):
    #import keras
    from keras.models import load_model
    
    m=load_model(filename)
    
    fixLayersContaining(m, fixOnlyContaining)
                
    return m

def load_model(filename):
    from keras.models import load_model
    
    model=load_model(filename, custom_objects=custom_objs)
    
    return model

def apply_weights_where_possible(target_model, weight_model):
    
    for layer_a in target_model.layers:
        for layer_b in weight_model.layers:
            if layer_a.name == layer_b.name:
                try:
                    layer_a.set_weights(layer_b.get_weights()) 
                except:  
                    print('unable to copy weights for layer ',  layer_a.name)
                    print(layer_a.weights,'\n',layer_b.weights)
    
    
    return target_model
